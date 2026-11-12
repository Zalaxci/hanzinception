import { blockUntilReady, jit, nn, numpy as np, random, tree, valueAndGrad, type OwnedFunction } from "@jax-js/jax"
import { adam, applyUpdates, type GradientTransformation } from "@jax-js/optax"
import { hanziList } from "./HanziReader"

type PredictorFunction = (w: np.Array[], x: np.Array) => np.Array
type PredictAndAutodiffFunc = (w: np.Array[], x: np.Array, y: np.Array) => [np.Array, np.Array[]]
export interface TrainMetrics {
    iepoch: number
    ibatch: number
    loss: number
}
interface TestResults {
    accuracy: number
    loss: number
}
type MultipleModelTestResults = Record<number, TestResults>

class NeuralNetworkOutput {
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
export class NeuralNetworkModel {
    _modelStructure: Array<number> // Array length of each layer, including input & output
    _layerWeights: np.Array[] // Weight matrices to multiply output of each layer to calculate input to the next layer's activation function and shift vectors to add after matrix multiplication at each layer
    _jitPredictor: OwnedFunction<PredictorFunction>
    constructor(modelStructure: Array<number>, predictor: PredictorFunction) {
        this._modelStructure = modelStructure; // Initialize model structure
        const nlayers = modelStructure.length - 1; // Layer weights & "shift" vectors for every layer but input
        this._layerWeights = new Array(2*nlayers);
        this._jitPredictor = jit(predictor); // Predictor function - as this is JIT compiled for efficiency it can't simply be a method
    }
    // Since the ownership of the weights and the predictor belongs to the object, i.e. we don't want it to be consumed and thus freed on a method call, we ought to have a destructor to free it
    destructor() {
        tree.dispose(this._layerWeights); // Use "tree.dispose" to dispose many Jax variables at once; for arrays equivalent to iterating over it & disposing each item
        this._jitPredictor.dispose();
    }
    // Use "valueAndGrad" as provided by Jax to create function that evaluates both the predictor's output & its derivatives (gradient) with respect to weights
    _getAutodifPredictor(): PredictAndAutodiffFunc {
        // !!! We need to keep track of how autodifferentiation works & memory management to avoid memory leaks !!!
        return valueAndGrad((w: np.Array[], input: np.Array, labels: np.Array): np.Array => {
            // Generate predictions, passing in the weights from the *1st function argument* -> this can't be a refernce to a class property, as "valueAndGrad" can only differentiate over function arguments! (argument 0 by default)
            const output = new NeuralNetworkOutput(
                // Here we can use the keyword "this" to access the model structure & JIT-compiled predictor function as Jax doesn't change the context
                this._modelStructure[this._modelStructure.length - 1], // Number of classes = length of last (output) layer
                this._jitPredictor(w, input),
            );
            // Calculate & store negative log-likelihood loss
            const loss = output.loss(labels);
            // !!! IMPORTANT: Free the output
            output.destructor();
            return loss;
        });
    }
    async initRandom() {
        const totalLayers = this._modelStructure.length // Total layer count, including input & output
        const nlayers = totalLayers - 1; // Number of hidden layers plus output
        const keys = random.split(random.key(Date.now()), 2*nlayers)
        const keysIterator = keys[Symbol.iterator]()
        for (let i = 0; i < nlayers; i++) {
            const bound = 1 / Math.sqrt(this._modelStructure[i])
            // Matrix right-multiplies row vector, so its number of rows is the size of the starting (i-th) layer and number of columns is the size of the (i+1)th layer
            this._layerWeights[2*i] = random.uniform(
                keysIterator.next().value,
                [this._modelStructure[i], this._modelStructure[i+1]],
                { minval: -bound, maxval: bound }
            )
            // Shift amount is added after matrix multiplication is applied and has size of (i+1)th layer
            this._layerWeights[2*i + 1] = random.uniform(
                keysIterator.next().value,
                [this._modelStructure[i+1]],
                { minval: -bound, maxval: bound }
            )
        }
    }
    predict(input: np.Array): NeuralNetworkOutput {
        return new NeuralNetworkOutput(
            this._modelStructure[this._modelStructure.length - 1],
            this._jitPredictor(tree.ref(this._layerWeights), input)
        )
    }
    async* train(inputIterable: AsyncGenerator<np.Array>, labelsIterable: AsyncGenerator<np.Array>, nbatches: number, nepochs: number, optimizerOrLearningRate: GradientTransformation | number): AsyncGenerator<TrainMetrics> {
        const optimizer = typeof optimizerOrLearningRate === "number"? adam(optimizerOrLearningRate) : optimizerOrLearningRate;
        const evaluateValueAndGradientAtOnce = this._getAutodifPredictor();
        let updates: np.Array[] = [];
        let optState = optimizer.init(tree.ref(this._layerWeights));
        // Use try; finally to ensure memory is cleared :)
        try {
            for (let iepoch = 0; iepoch < nepochs; iepoch++) {
                let curr_input = await inputIterable.next();
                let curr_labels = await labelsIterable.next();
                for (let ibatch = 0; ibatch < nbatches; ibatch++) {
                    const [lossVal, derivativeOverWeights] = evaluateValueAndGradientAtOnce(tree.ref(this._layerWeights), curr_input.value.ref, curr_labels.value.ref);
                    if (ibatch % 10 === 0) {
                        const lossAsNumber = await lossVal.jsAsync();
                        if (isNaN(lossAsNumber)) {
                            console.error("LOSS IS `NOT A NUMBER` (NaN) - ABORTING!");
                            const prediction = this.predict(curr_input.value);
                            console.log("Predicted log probabilities: ", await prediction._logits.jsAsync());
                            console.log("Actual class is: ", await curr_labels.value.jsAsync());
                            return;
                        }
                        yield { iepoch, ibatch, loss: lossAsNumber };
                        // If loss smaller than 0.03 return; memory will be cleared in the "finally" block
                        if (lossAsNumber < 0.03) return;
                    } else lossVal.dispose();
                    curr_input.value.dispose();
                    curr_labels.value.dispose();
                    [updates, optState] = optimizer.update(derivativeOverWeights, optState);
                    this._layerWeights = applyUpdates(this._layerWeights, updates);
                    if (ibatch % 6 === 0)
                        [curr_input, curr_labels] = await Promise.all([ inputIterable.next(), labelsIterable.next(), blockUntilReady(this._layerWeights) ]);
                    else
                        [curr_input, curr_labels] = await Promise.all([ inputIterable.next(), labelsIterable.next() ]);
                }
            }
        } catch(e) {
            console.warn("training error: " + e);
        } finally {
            console.log("Clearing training function memory!");
            tree.dispose(optState);
        }
    }
}
export class HanziRecognizer3LayerMLP extends NeuralNetworkModel {
    constructor() {
        super(
            [2500, 500, 400, 300, 200, 200, hanziList.length],
            (w: np.Array[], x: np.Array): np.Array => {
                // Use np.dot as JAX.js' creators use that for MNIST, and matmul goes over array indices as string which might be slower
                const z1 = np.dot(x, w[0]).add(w[1])
                // Hidden layers use ReLU
                const a1 = nn.relu(z1)
                const z2 = np.dot(a1, w[2]).add(w[3])
                const a2 = nn.relu(z2)
                const z3 = np.dot(a2, w[4]).add(w[5])
                const a3 = nn.relu(z3)
                const z4 = np.dot(a3, w[6]).add(w[7])
                const a4 = nn.relu(z4)
                const z5 = np.dot(a4, w[8]).add(w[9])
                const a5 = nn.relu(z5)
                // Final layer uses softmax
                const z6 = np.dot(a5, w[10]).add(w[11])
                return nn.logSoftmax(z6)
            }
        )
    }
}
export class NeuralNetworkManager {
    _enabledModels: Set<number> = new Set()
    _models: Array<NeuralNetworkModel> = []
    // Clear memory for each model (this class has no Jax variables directly as parameters to clear)
    destructor() {
        for (let model of this._models)
            model.destructor();
    }
    addModel(model: NeuralNetworkModel) {
        this._models.push(model)
    }
    enableModel(modelId: number) {
        this._enabledModels.add(modelId)
    }
    disableModel(modelId: number) {
        this._enabledModels.delete(modelId)
    }
    async initRandom() {
        // Initialize each model with a random key - pass in current time in milliseconds in "random.key", though I'm not sure if this is necessary or even capable of guaranteeing different random values per model
        for (let modelId of this._enabledModels)
            await this._models[modelId].initRandom();
    }
    async train(inputIterable: AsyncGenerator<np.Array>, labelsIterable: AsyncGenerator<np.Array>, nbatches: number, nepochs: number, learningRate: number = 0.008) {
        const adamOptimizer = adam(learningRate);
        for (let imodel of this._enabledModels)
            await this._models[imodel].train(inputIterable, labelsIterable, nbatches, nepochs, adamOptimizer);
    }
    async test(input: np.Array, labels: np.Array): Promise<MultipleModelTestResults> {
        const results: MultipleModelTestResults = {};
        for (let modelId of this._enabledModels) {
            let prediction = this._models[modelId].predict(input.ref);
            results[modelId] = {
                accuracy: await prediction.accuracy(labels.ref),
                loss: await prediction.loss(labels.ref).jsAsync()
            };
            prediction.destructor();
        }
        input.dispose();
        labels.dispose();
        return results;
    }
    predict(modelId: number, input: np.Array): NeuralNetworkOutput {
        return this._models[modelId].predict(input)
    }
}