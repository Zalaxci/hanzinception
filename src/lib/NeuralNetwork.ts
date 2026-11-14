import { blockUntilReady, jit, lax, nn, numpy as np, random, tree, valueAndGrad, type OwnedFunction } from "@jax-js/jax"
import { adamw, applyUpdates, type GradientTransformation } from "@jax-js/optax"
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
    // Return logits as JS array, freeing up Jax memory in the process
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
            const output = new NeuralNetworkOutput(
                // Here we can use the keyword "this" to access the model structure & JIT-compiled predictor function as Jax doesn't change the context
                this._noutputs, // Number of classes = length of last (output) layer
                this._jitPredictor(w, input),
            );
            // Calculate & store negative log-likelihood loss
            const loss = output.loss(labels);
            // !!! IMPORTANT: Free the output
            output.destructor();
            return loss;
        });
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
            )
            // Shift amount is added after matrix multiplication is applied and has size of (i+1)th layer
            this._weights[2*i + 1] = random.uniform(
                keysIterator.next().value,
                this._weightShape[2*i + 1],
                { minval: -bound, maxval: bound }
            )
        }
    }
    predict(input: np.Array): NeuralNetworkOutput {
        return new NeuralNetworkOutput(
            this._noutputs,
            this._jitPredictor(tree.ref(this._weights), input)
        )
    }
    async* train(inputs: AsyncGenerator<np.Array>, labels: np.Array, nbatches: number, nepochs: number, learningRate: number): AsyncGenerator<TrainMetrics> {
        const optimizer = adamw(learningRate, { mask: null, weightDecay: 0.01 });
        const evaluateValueAndGradientAtOnce = this._getAutodifPredictor();
        let updates: np.Array[] = [];
        let optState = optimizer.init(tree.ref(this._weights));
        // Use try; finally to ensure memory is cleared :)
        try {
            for (let iepoch = 0; iepoch < nepochs; iepoch++) {
                let curr_input = await inputs.next();
                for (let ibatch = 0; ibatch < nbatches; ibatch++) {
                    const [lossVal, derivativeOverWeights] = evaluateValueAndGradientAtOnce(tree.ref(this._weights), curr_input.value.ref, labels.ref);
                    if (ibatch % 20 === 0) {
                        const lossAsNumber = await lossVal.jsAsync();
                        if (isNaN(lossAsNumber)) {
                            console.error("LOSS IS `NOT A NUMBER` (NaN) - ABORTING!");
                            const prediction = this.predict(curr_input.value);
                            console.log("Predicted log probabilities: ", await prediction._logits.jsAsync());
                            console.log("Actual class is: ", await labels.ref.jsAsync());
                            return;
                        }
                        yield { iepoch, ibatch, loss: lossAsNumber };
                        // If loss smaller than 0.05 return; memory will be cleared in the "finally" block
                        if (lossAsNumber < 0.05) return;
                    } else lossVal.dispose();
                    curr_input.value.dispose();
                    [updates, optState] = optimizer.update(derivativeOverWeights, optState, tree.ref(this._weights));
                    this._weights = applyUpdates(this._weights, updates);
                    if (ibatch % 5 === 0)
                        [ curr_input ] = await Promise.all([ inputs.next(), blockUntilReady(this._weights) ]);
                    else
                        curr_input = await inputs.next();
                }
            }
        } catch(e) {
            console.warn("training error: " + e);
        } finally {
            console.log("Clearing training function memory!");
            labels.dispose();
            tree.dispose(optState);
        }
    }
}
export class HanziRecognizerMLP extends NeuralNetworkModel {
    constructor() {
        super(
            hanziList.length,
            // Weight shapes; odd index numbers correspond to matrices we multiply by, even index numbers to offset vectors for each set of neurons
            [[2500, 500], [500], [500, 400], [400], [400, 300], [300], [300, 200], [200], [200, 200], [200], [200, hanziList.length], [hanziList.length]],
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
        );
    }
}
export class HanziRecognizerCNN extends NeuralNetworkModel {
    constructor() {
        super(
            hanziList.length,
            [
                [16, 1, 3, 3],
                [16, 1, 1],
                [64, 16, 3, 3],
                [64, 1, 1],
                [128, 64, 3, 3],
                [128, 1, 1],
                [256, 128, 3, 3],
                [256, 1, 1],
                [256, hanziList.length],
                [hanziList.length],
            ],
            (w: np.Array[], x: np.Array): np.Array => {
                x = x.reshape([-1, 1, 50, 50]); // x is Nx1x50x50
                const conv1 = lax.convGeneralDilated(x, w[0], [1, 1], "VALID").add(w[1]); // 3x3 convolution => conv1 is Nx16x48x48
                const z1 = lax.reduceWindow(conv1, np.max, [2, 2], [2, 2]); // 2x2 max pooling => z1 is Nx16x24x24
                const a1 = nn.relu(z1); // ReLU activation function
                const conv2 = lax.convGeneralDilated(a1, w[2], [1, 1], "VALID").add(w[3]); // 3x3 convolution => conv2 is Nx64x22x22
                const z2 = lax.reduceWindow(conv2, np.max, [2, 2], [2, 2]); // 2x2 max pooling => z2 is Nx64x11x11
                const a2 = nn.relu(z2); // ReLU activation function (a2 is Nx64x11x11)
                const conv3 = lax.convGeneralDilated(a2, w[4], [1, 1], "VALID").add(w[5]); // 3x3 convolution => conv2 is Nx128x9x9
                const z3 = lax.reduceWindow(conv3, np.max, [3, 3], [3, 3]); // 3x3 max pooling => z3 is 128x3x3
                const a3 = nn.relu(z3); // ReLU activation function (a2 is Nx128x3x3)
                const conv4 = lax.convGeneralDilated(a3, w[6], [1, 1], "VALID").add(w[7]); // 3x3 convolution => conv2 is Nx256x1x1
                const a4 = nn.relu(conv4);
                const a4flat = a4.reshape([-1, 256]);
                const z5 = np.dot(a4flat, w[8]).add(w[9]);
                return nn.logSoftmax(z5);
                
            }
        );
        this._initialWeightBound[0] = 1 / 3;
        this._initialWeightBound[1] = 1 / (3 * Math.sqrt(16));
        this._initialWeightBound[2] = 1 / (3 * Math.sqrt(64));
        this._initialWeightBound[3] = 1 / (3 * Math.sqrt(128));
    }
}
// Pure CNN: ONLY 3x3 convolutions - no pooling or traditional // MLP layers in the end!
// See: https://arxiv.org/html/1412.6806v3
// 8 total layers, 7 hidden
// Only using convolutional layers seems to result in overfitting :(
export class HanziRecognizerPureCNN extends NeuralNetworkModel {
    constructor() {
        super(
            hanziList.length,
            [
                [16, 1, 3, 3],
                [16, 1, 1],
                [16, 16, 3, 3],
                [16, 1, 1],
                [32, 16, 3, 3],
                [32, 1, 1],
                [32, 32, 3, 3],
                [32, 1, 1],
                [32, 32, 3, 3],
                [32, 1, 1],
                [64, 32, 3, 3],
                [64, 1, 1],
                [128, 64, 3, 3],
                [128, 1, 1],
                [hanziList.length, 128, 3, 3],
                [hanziList.length, 1, 1],
            ],
            (w: np.Array[], x: np.Array) => {
                x = x.reshape([-1, 1, 50, 50]); // x is Nx1x50x50 (50x50 pixels, 1 channel)
                const conv1 = lax.convGeneralDilated(x, w[0], [1, 1], "VALID").add(w[1]); // 3x3 convolution with stride 1 => shape Nx16x48x48 (48x48 pixels, 8 channels)
                const a1 = nn.relu(conv1);
                const conv2 = lax.convGeneralDilated(a1, w[2], [1, 1], "VALID").add(w[3]); // 3x3 convolution with stride 1 => shape Nx16x46x46 (46x46 pixels, 8 channels)
                const a2 = nn.relu(conv2);
                const conv3 = lax.convGeneralDilated(a2, w[4], [2, 2], "VALID").add(w[5]); // 3x3 convolution with stride 2 => shape Nx32x22x22 (23x23 pixels, 16 channels)
                const a3 = nn.relu(conv3);
                const conv4 = lax.convGeneralDilated(a3, w[6], [1, 1], "VALID").add(w[7]); // 3x3 convolution with stride 1 => shape Nx32x20x20 (21x21 pixels, 16 channels)
                const a4 = nn.relu(conv4);
                const conv5 = lax.convGeneralDilated(a4, w[8], [1, 1], "VALID").add(w[9]); // 3x3 convolution with stride 1 => shape Nx32x18x18 (21x21 pixels, 16 channels)
                const a5 = nn.relu(conv5);
                const conv6 = lax.convGeneralDilated(a5, w[10], [2, 2], "VALID").add(w[11]); // 3x3 convolution with stride 2 => shape Nx64x8x8 (10x10 pixels, 32 channels)
                const a6 = nn.relu(conv6);
                const conv7 = lax.convGeneralDilated(a6, w[12], [2, 2], "VALID").add(w[13]); // 3x3 convolution with stride 2 => shape Nx128x3x3 (4x4 pixels, 64 channels)
                const a7 = nn.relu(conv7);
                const conv8 = lax.convGeneralDilated(a7, w[14], [2, 2], "VALID").add(w[15]); // 3x3 convolution => shape NxNHANZIx1x1 (1x1 pixel, 1 channel per hanzi)
                const conv8flat = conv8.reshape([-1, hanziList.length]); // NxNHANZI
                return nn.logSoftmax(conv8flat);
            }
        )
        this._initialWeightBound[0] = 1 / 3;
        this._initialWeightBound[1] = 1 / (3 * Math.sqrt(16));
        this._initialWeightBound[2] = 1 / (3 * Math.sqrt(16));
        this._initialWeightBound[3] = 1 / (3 * Math.sqrt(32));
        this._initialWeightBound[4] = 1 / (3 * Math.sqrt(32));
        this._initialWeightBound[5] = 1 / (3 * Math.sqrt(32));
        this._initialWeightBound[6] = 1 / (3 * Math.sqrt(64));
        this._initialWeightBound[7] = 1 / (3 * Math.sqrt(128));
    }
}