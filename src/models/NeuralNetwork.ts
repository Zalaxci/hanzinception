import { blockUntilReady, jit, lax, nn, numpy as np, random, tree, valueAndGrad, type OwnedFunction } from "@jax-js/jax"
import { adamw, applyUpdates } from "@jax-js/optax"

export type RecursiveArray<T> = T | RecursiveArray<T>[]

type PredictorFunction = (w: np.Array[], x: np.Array) => np.Array
type PredictAndAutodiffFunc = (w: np.Array[], x: np.Array, y: np.Array) => [np.Array, np.Array[]]

class NNPredictionModel {
    _nclasses: number
    _logits: np.Array // Logit matrix with each row corresponding to an example and each column corresponding to the log-probability the model gives for each output
    constructor(nclasses: number, p: np.Array) {
        this._nclasses = nclasses
        this._logits = p
    }
    // Since the ownership of "logits" belongs to the object, i.e. we don't want it to be consumed and thus freed on a method call, we ought to have a destructor to free it
    destructor() {
        this._logits.dispose();
    }
    // Return probabilities as JS array, freeing up Jax memory in the process
    realizeDataAndDestroy(): Promise<number[]> {
        return np.exp(this._logits).data();
    }
    mostLikelyOutputs(): np.Array {
        // Return most likely output or outputs, effectively the "argmax", i.e. the column index with greatest value for each row
        return np.argmax(this._logits.ref, 1);
    }
    loss(labels: np.Array): np.Array {
        const nexamples = labels.shape[0]
        // Get negative log-likehood loss:; For each example muliplying by one-hot encoded labels selects the log-probability of the correct output
        return this._logits
                   .ref
                   .mul(nn.oneHot(labels, this._nclasses)) // element-wise multiplication
                   .sum() // sum over all examples
                   .mul(-1 / nexamples); // divide by total number of examples to get the average and multiply by -1 to make positive (the higher the loss the worse)
    }
    async accuracy(jaxLabels: np.Array): Promise<number> {
        const nexamples = jaxLabels.shape[0];
        const labels = await jaxLabels.data();
        const predictedLabels = await this.mostLikelyOutputs().data();
        let correctGuesses = 0;
        for (let i = 0; i < nexamples; i++)
            if (labels[i] === predictedLabels[i])
                correctGuesses++;
        return correctGuesses/nexamples;
    }
}
export type { NNPredictionModel };
export interface NNTrainInput {
    labels: np.Array
    data: np.Array
}
export interface NNTrainMetrics {
    iepoch: number
    ibatch: number
    loss: number
}
export class NeuralNetworkModel {
    _noutputs: number
    _weightShape: number[][] // Matrix shape (similar to numpy) for each weight
    _initialWeightBound: Record<number, number>
    _weights: np.Array[] // Weight matrices to multiply output of each layer to calculate input to the next layer's activation function and shift vectors to add after matrix multiplication at each layer
    _jitPredictor: OwnedFunction<PredictorFunction> // Predictor function - as this is JIT compiled for efficiency it can't simply be a method
    constructor(noutputs: number, weightShape: number[][], predictor: PredictorFunction) {
        this._noutputs = noutputs;
        this._weightShape = weightShape;
        this._initialWeightBound = {};
        this._weights = new Array(weightShape.length);
        this._jitPredictor = jit(predictor);
    }
    // Since the ownership of the weights and the predictor belongs to the object, i.e. we don't want it to be consumed and thus freed on a method call, we ought to have a destructor to free it
    destructor() {
        tree.dispose(this._weights); // Use "tree.dispose" to dispose many Jax variables at once; for arrays equivalent to iterating over it & disposing each item
        this._jitPredictor.dispose();
    }
    // Use "valueAndGrad" as provided by Jax to create function that evaluates both the predictor's output & its derivatives (gradient) with respect to weights
    _getAutodifPredictor(): PredictAndAutodiffFunc {
        // !!! We need to keep track of how autodifferentiation works & memory management to avoid memory leaks !!!
        return valueAndGrad((w: np.Array[], input: np.Array, labels: np.Array): np.Array => {
            // Generate predictions, passing in the weights from the *1st function argument* -> this can't be a refernce to a class property, as "valueAndGrad" can only differentiate over function arguments! (argument 0 by default)
            const prediction = new NNPredictionModel(
                // Here we can use the keyword "this" to access the model structure & JIT-compiled predictor function as Jax doesn't change the context
                this._noutputs, // Number of classes = length of last (output) layer
                this._jitPredictor(w, input),
            );
            // Calculate & store negative log-likelihood loss
            const loss = prediction.loss(labels);
            // !!! IMPORTANT: Free the output
            prediction.destructor();
            return loss;
        });
    }
    async getWeights(): Promise<RecursiveArray<number>> {
        return await Promise.all(this._weights.map(w => w.ref.jsAsync() as Promise<RecursiveArray<number>>));
    }
    async loadWeights() {
        const response = await fetch("weights.json");
        const weights = await response.json() as RecursiveArray<number>[];
        this._weights = weights.map(w => np.array(w));
    }
    // Init random values - this only works for MLPs, not convolutional neural networks
    async initRandom() {
        const nlayers = this._weightShape.length / 2;
        const keys = random.split(random.key(Date.now()), 2*nlayers);
        const keysIterator = keys[Symbol.iterator]();
        for (let i = 0; i < nlayers; i++) {
            const bound = this._initialWeightBound[i] || 1 / Math.sqrt(this._weightShape[2*i][0]);
            // Matrix right-multiplies row vector, so its number of rows is the size of the starting (i-th) layer and number of columns is the size of the (i+1)th layer
            this._weights[2*i] = random.uniform(
                keysIterator.next().value,
                this._weightShape[2*i],
                { minval: -bound, maxval: bound }
            ).astype(np.float16);
            // Shift amount is added after matrix multiplication is applied and has size of (i+1)th layer
            this._weights[2*i + 1] = random.uniform(
                keysIterator.next().value,
                this._weightShape[2*i + 1],
                { minval: -bound, maxval: bound }
            ).astype(np.float16);
        }
    }
    predict(input: np.Array): NNPredictionModel {
        return new NNPredictionModel(
            this._noutputs,
            this._jitPredictor(tree.ref(this._weights), input)
        )
    }
    async* train(inputs: AsyncGenerator<NNTrainInput>, nbatches: number, nepochs: number, learningRate: number, cutoffLoss: number = 0): AsyncGenerator<NNTrainMetrics> {
        const optimizer = adamw(learningRate, { mask: null, weightDecay: 0.005 });
        const evaluateValueAndGradientAtOnce = this._getAutodifPredictor();
        let updates: np.Array[] = [];
        let optState = optimizer.init(tree.ref(this._weights));
        let ibatch: number = 0, iepoch: number = 0;
        // Use try; finally to ensure memory is cleared :)
        try {
            for (iepoch = 0; iepoch < nepochs; iepoch++) {
                let curr_input = await inputs.next() as { value: NNTrainInput };
                for (ibatch = 0; ibatch < nbatches; ibatch++) {
                    const [lossVal, derivativeOverWeights] = evaluateValueAndGradientAtOnce(tree.ref(this._weights), curr_input.value.data.ref, curr_input.value.labels.ref);
                    if ((ibatch + 1) % 10 === 0) {
                        const lossAsNumber = await lossVal.jsAsync();
                        if (isNaN(lossAsNumber)) {
                            console.error("LOSS IS `NOT A NUMBER` (NaN) - ABORTING!");
                            const prediction = this.predict(curr_input.value.data);
                            console.log("Predicted log probabilities: ", await prediction.realizeDataAndDestroy());
                            console.log("Actual class is: ", await curr_input.value.labels.jsAsync());
                            return;
                        }
                        yield { iepoch: iepoch + 1, ibatch: ibatch + 1, loss: lossAsNumber };
                        if (cutoffLoss > 0 && lossAsNumber <= cutoffLoss) return;
                    } else lossVal.dispose();
                    // Dispose labels & data memory
                    curr_input.value.labels.dispose();
                    curr_input.value.data.dispose();
                    [updates, optState] = optimizer.update(derivativeOverWeights, optState, tree.ref(this._weights));
                    this._weights = applyUpdates(this._weights, updates);
                    if (ibatch % 8 === 0)
                        [ curr_input ] = await Promise.all([ inputs.next(), blockUntilReady(this._weights) ]);
                    else
                        curr_input = await inputs.next();
                }
            }
            await blockUntilReady(this._weights);
        } catch(e) {
            console.error(`Training error ${e} on batch ${ibatch}, epoch ${iepoch}`);
        } finally {
            console.log("Clearing training function memory!");
            tree.dispose(optState);
        }
    }
}
