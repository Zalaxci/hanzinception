import { blockUntilReady, jit, nn, numpy as np, random, tree, valueAndGrad, type OwnedFunction } from "@jax-js/jax"
import { adam, applyUpdates, type GradientTransformation } from "@jax-js/optax"

type PredictorFunction = (w: np.Array[], x: np.Array) => np.Array
type PredictAndAutodiffFunc = (w: np.Array[], x: np.Array, y: np.Array) => [np.Array, np.Array[]]
interface TestResults {
    accuracy: number
    loss: number
}
type MultipleModelTestResults = Record<number, TestResults>

// Pubsub
// interface ChannelSubscriber<MSGType> {
//     onEvent(producerId: number, msg: MSGType): void
//     onFinished(producerId: number, success: boolean): void
//     onAllFinished(success: boolean): void
//     onError(producerId: number, err: Error): void
// }
// class Channel<MSGType> {
//     subscribers: ChannelSubscriber<MSGType>[] = []
//     subscribe(subscriber: ChannelSubscriber<MSGType>) {
//         this.subscribers.push(subscriber)
//     }

// }
// Create array with indices in random order for array of length "range" (from 0 to range - 1) utilizing only one array, the one returned, and no memory reallocation
function randomIndices(range: number): np.Array {
    // Create array with length "range"
    let indices = new Array(range)
    // Repeat for all indices
    for (let i = 0; i < range; i++) {
        // Peak random position to place
        let posToAdd = Math.floor(Math.random() * range)
        // If already occupied, move to next position, cycling back to 0 once last index is reached, to find where to place
        while (indices[posToAdd] !== undefined) {
            if (posToAdd === range - 1) posToAdd = 0
            else posToAdd++
        }
        // Place "i", i.e. the current index, in specified position
        indices[posToAdd] = i
    }
    // Convert to np.array which enables faster computation
    return np.array(indices, { dtype: np.uint32 })
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
    async initRandom(key: np.Array) {
        const totalLayers = this._modelStructure.length // Total layer count, including input & output
        const nlayers = totalLayers - 1; // Number of hidden layers plus output
        const keys = random.split(key, 2*nlayers)
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
    async train(input: np.Array, labels: np.Array, batchSize: number, optimizer: GradientTransformation) {
        const nexamples = input.shape[0]
        const nbatches = Math.ceil(nexamples / batchSize)
        const evaluateValueAndGradientAtOnce = this._getAutodifPredictor()
        let updates: np.Array[] = []
        let optState = optimizer.init(tree.ref(this._layerWeights))
        // Use try; finally to ensure memory is cleared :)
        try {
            for (let iepoch = 0; iepoch < 10; iepoch++) {
                const epochExampleIndices = randomIndices(nexamples)
                try {
                    for (let ibatch = 0; ibatch < nbatches; ibatch++) {
                        console.log(`Batch ${ibatch} of ${nbatches}, batch size ${batchSize} out of ${nexamples} examples`)
                        const batchExampleIndices = epochExampleIndices.ref.slice([ibatch * batchSize, (ibatch + 1) * batchSize])
                        const curr_input = input.ref.slice(batchExampleIndices.ref)
                        const curr_labels = labels.ref.slice(batchExampleIndices)
                        const [lossVal, derivativeOverWeights] = evaluateValueAndGradientAtOnce(tree.ref(this._layerWeights), curr_input, curr_labels)
                        ;[updates, optState] = optimizer.update(derivativeOverWeights, optState)
                        this._layerWeights = applyUpdates(this._layerWeights, updates)
                        await blockUntilReady(this._layerWeights)
                        const lossAsNumber = await lossVal.jsAsync() as number
                        console.log(iepoch * nbatches + ibatch, lossAsNumber)
                        // If loss smaller than 0.01 return; memory will be cleared in the "finally" block
                        if (lossAsNumber < 0.01) return;
                    }
                } finally {
                    console.log("Clearing training epoch memory!");
                    epochExampleIndices.dispose();
                }
            }
        } finally {
            console.log("Clearing training function memory!");
            input.dispose();
            labels.dispose();
            tree.dispose(optState);
        }
    }
}
export class MNISTDualLayerNN extends NeuralNetworkModel {
    constructor() {
        super(
            [784, 200, 100, 10],
            (w: np.Array[], x: np.Array): np.Array => {
                // Flatten each image to 784-size row vector; use "-1" to allow any number of rows so that any number of images can be passed in at once
                x = x.reshape([-1, 784])
                // Use np.dot as JAX.js' creators use that for MNIST, and matmul goes over array indices as string which might be slower
                const z1 = np.dot(x, w[0]).add(w[1])
                // Hidden layers use softplus
                const a1 = nn.softplus(z1)
                const z2 = np.dot(a1, w[2]).add(w[3])
                const a2 = nn.softplus(z2)
                // Final layer uses softmax
                const z3 = np.dot(a2, w[4]).add(w[5])
                return nn.logSoftmax(z3)
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
            await this._models[modelId].initRandom(random.key(Date.now()))
    }
    async train(input: np.Array, labels: np.Array, batchSize: number, learningRate: number = 0.008) {
        const adamOptimizer = adam(learningRate);
        for (let imodel of this._enabledModels)
            await this._models[imodel].train(input.ref, labels.ref, batchSize, adamOptimizer);
        input.dispose();
        labels.dispose();
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