import { jit, nn, numpy as np, random, type OwnedFunction } from "@jax-js/jax"

type PredictorFunction = (w: Array<np.Array>, b: Array<np.Array>, x: np.Array) => np.Array

class NeuralNetworkOutput {
    _nclasses: number
    _logits: np.Array // Logit matrix with each row corresponding to an example and each column corresponding to the log-probability the model gives for each output
    constructor(nclasses: number, p: np.Array) {
        this._nclasses = nclasses
        this._logits = p
    }
    mostLikelyOutputs(): np.Array {
        // Return most likely output or outputs, effectively the "argmax", i.e. the column index with greatest value for each row
        return np.argmax(this._logits, 1)
    }
    loss(labels: np.Array): np.Array {
        const nexamples = labels.shape[0]
        // Get negative log-likehood loss: For each example muliplying by one-hot encoded labels selects the log-probability of the correct output
        return this._logits
                   .mul(nn.oneHot(labels, this._nclasses)) // element-wise multiplication
                   .sum() // sum over all examples
                   .mul(-1 / nexamples) // divide by total number of examples to get the average and multiply by -1 to make positive (the higher the loss the worse)
    }
    accuracy(labels: np.Array): np.Array {
        const nexamples = labels.shape[0]
        // Sum over array which is one where predictions equal labels and divide by number of examples
        return this.mostLikelyOutputs()
                   .equal(labels)
                   .astype(np.uint32)
                   .sum()
                   .mul(1 / nexamples)
    }
}
class NeuralNetworkModel {
    _modelStructure: Array<number> // Array length of each layer, including input & output
    _layerWeights: Array<np.Array> // Weight matrices to multiply output of each layer to calculate input to the next layer's activation function
    _layerShifts: Array<np.Array> // Shift vectors to add after matrix multiplication at each layer
    _jitPredictor: OwnedFunction<PredictorFunction>
    constructor(modelStructure: Array<number>, predictor: PredictorFunction) {
        this._modelStructure = modelStructure // Initialize model structure
        this._layerWeights = new Array(modelStructure.length - 1) // Layer weights for every layer but input
        this._layerShifts = new Array(modelStructure.length - 1) // Layer "shift" vectors for every layer but input
        this._jitPredictor = jit(predictor) // Predictor function, JIT compiled
    }
    async initRandom(key: np.Array) {
        const totalLayers = this._modelStructure.length // Total layer count, including input & output
        const keys = await random.split(key, 2*totalLayers).data()
        const nlayers = totalLayers - 1; // Number of hidden layers plus output
        for (let i = 0; i < nlayers; i++) {
            // Matrix right-multiplies row vector, so its number of rows is the size of the starting (i-th) layer and number of columns is the size of the (i+1)th layer
            this._layerWeights[i] = random.uniform(keys[2*i], [this._modelStructure[i], this._modelStructure[i+1]])
            // Shift amount is added after matrix multiplication is applied and has size of (i+1)th layer
            this._layerShifts[i] = random.uniform(keys[2*i + 1], [this._modelStructure[i+1]])
        }
    }
    predict(input: np.Array): NeuralNetworkOutput {
        return new NeuralNetworkOutput(
            this._modelStructure[this._modelStructure.length - 1],
            this._jitPredictor(this._layerWeights, this._layerShifts, input)
        )
    }
}
class MNISTDualLayerNN extends NeuralNetworkModel {
    constructor() {
        super(
            [784, 200, 10],
            (w: Array<np.Array>, b: Array<np.Array>, x: np.Array): np.Array => {
                // Flatten each image to 784-size row vector; use "-1" to allow any number of rows so that any number of images can be passed in at once
                x = x.reshape([-1, 784])
                // Use np.dot as JAX.js' creators use that for MNIST, and matmul goes over array indices as string which might be slower
                const z1 = np.dot(x, w[0]).add(b[0])
                // Hidden layer uses ReLU
                const a1 = nn.relu(z1)
                // Final layer uses softmax
                const z2 = np.dot(a1, w[1])
                return nn.logSoftmax(z2)
            }
        )
    }
}
class NeuralNetworkManager {
    _enabledModels: Set<number> = new Set()
    _models: Array<NeuralNetworkModel> = []
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
    train(input: np.Array, labels: np.Array) {}
    test(input: np.Array, labels: np.Array) {
        // for (let modelId of this._enabledModels) {
        //     let prediction = this._models[modelId].predict(input)
        // }
    }
    predict(modelId: number, input: np.Array): NeuralNetworkOutput {
        return this._models[modelId].predict(input)
    }
}