import { numpy as np } from "@jax-js/jax";
import { getHanzi, getImageSize, getTotalHanziCount, N_TEST_IMAGES, N_TRAIN_IMAGES, type HanziStorageModel } from "../models/HanziStorage";
import type { NeuralNetworkModel, NNTrainInput, NNTrainMetrics } from "../models/NeuralNetwork";

// Create array with number from 0 to N - 1 in random order utilizing only one array, the one returned, and no memory reallocation
function randomlyOrderNumbersUpToN(arr: number[]) {
    // Set entire array to -1
    for (let i = 0; i < arr.length; i++)
        arr[i] = -1;
    // Repeat for all numbers from 1 to N
    for (let i = 0; i < arr.length; i++) {
        // Pick random position to place
        let posToAdd = Math.floor(Math.random() * arr.length);
        // If already occupied, move to next position, cycling back to 0 once last index is reached, to find where to place
        while (arr[posToAdd] !== -1) {
            if (posToAdd === arr.length - 1) posToAdd = 0;
            else posToAdd++;
        }
        // Place "i", i.e. the current index, in specified position
        arr[posToAdd] = i;
    }
}

export interface NNTrainingConfiguration {
    modelName: string
    neuralNet: NeuralNetworkModel
    batchSize: number
    learningRate: number
}
export class HanziRecognitionTrainer {
    #batchSize: number
    #nhanzi: number
    #imageSize: number
    #trainPixelBuffer: number[]
    #trainImageData: (number[] | undefined)[]
    #storage: HanziStorageModel
    #trainingConfigs: NNTrainingConfiguration[]
    constructor(storage: HanziStorageModel) {
        this.#storage = storage;
        this.#nhanzi = getTotalHanziCount();
        this.#imageSize = getImageSize();
        this.#batchSize = 0;
        this.#trainPixelBuffer = [];
        // 2D array with rows equal to hanzi number & columns the images per hanzi
        this.#trainImageData = new Array(this.#nhanzi*N_TRAIN_IMAGES);
        // Array of neural networks with their respective training configurations
        this.#trainingConfigs = [];
    }
    destructor() {
        for (let { neuralNet } of this.#trainingConfigs)
            neuralNet.destructor();
    }
    async #readTrainImageDataCached(iimg: number, bufferpos: number): Promise<void> {
        const npixels = this.#imageSize*this.#imageSize;
        if (!this.#trainImageData[iimg]) {
            console.log(`Image ${iimg} not in memory, loading from OPFS...`);
            const ihanzi = iimg % this.#nhanzi;
            const imgFileNum = (iimg - ihanzi)/this.#nhanzi + 1;
            const png = await this.#storage.readFile(getHanzi(ihanzi), imgFileNum, false);
            this.#trainImageData[iimg] = new Array(npixels);
            await png.decompressAndDecodePixels(this.#trainImageData[iimg]);
        }
        for (let ipixel = 0; ipixel < npixels; ipixel++)
            this.#trainPixelBuffer[bufferpos*npixels + ipixel] = this.#trainImageData[iimg][ipixel];
    }
    async *#createBatchGenerator(nepochs: number): AsyncGenerator<NNTrainInput> {
        const nbatches = Math.ceil(N_TRAIN_IMAGES * this.#nhanzi / this.#batchSize);
        const npixels = this.#imageSize*this.#imageSize;
        const epochIndices: number[] = new Array(N_TRAIN_IMAGES * this.#nhanzi);
        for (let iepoch = 0; iepoch < nepochs; iepoch++) {
            randomlyOrderNumbersUpToN(epochIndices);
            for (let ibatch = 0; ibatch < nbatches; ibatch++) {
                const batchIndices = epochIndices.slice(ibatch*this.#batchSize, (ibatch + 1)*this.#batchSize);
                await Promise.all(
                    batchIndices.map(this.#readTrainImageDataCached.bind(this))
                );
                const batchLabels = batchIndices.map(i => i % this.#nhanzi);
                yield {
                    labels: np.array(batchLabels, { dtype: np.uint32 }),
                    data: np.array(this.#trainPixelBuffer.slice(0, batchIndices.length*npixels), { dtype: np.float16, shape: [batchIndices.length, npixels]}),
                };
            }
        }
    }
    addConfigurations(...configs: NNTrainingConfiguration[]) {
        this.#trainingConfigs.push(...configs);
    }
    async initRandom() {
        for (let { neuralNet } of this.#trainingConfigs)
            await neuralNet.initRandom();
    }
    async *trainAll(nepochs: number, cutoffLoss: number = 0): AsyncGenerator<{iconfig: number, metrics: NNTrainMetrics}> {
        for (let [iconfig, config] of this.#trainingConfigs.entries()) {
            this.#batchSize = config.batchSize;
            const nbatches = Math.ceil(N_TRAIN_IMAGES * this.#nhanzi / this.#batchSize);
            // Temporary container for pixel data, capable of fitting batchSize images to avoid race condition when loading many images asynchronously
            this.#trainPixelBuffer = new Array(this.#batchSize*this.#imageSize*this.#imageSize);
            // Generate random batches of specific batch size
            const batchGenerator = this.#createBatchGenerator(nepochs);
            for await (let metrics of config.neuralNet.train(batchGenerator, nbatches, nepochs, config.learningRate, cutoffLoss))
                yield { iconfig, metrics };
        }
    }
    async *testAll(): AsyncGenerator<{iconfig: number, accuracy: number, loss: number}> {
        const npixels = this.#imageSize*this.#imageSize;
        const testPixelBuffer = new Array(N_TEST_IMAGES*npixels) as number[];
        const accuracies = new Float32Array(this.#trainingConfigs.length);
        const losses = new Float32Array(this.#trainingConfigs.length);
        for (let ihanzi = 0; ihanzi < this.#nhanzi; ihanzi++) {
            for (let iimg = 0; iimg < N_TEST_IMAGES; iimg++) {
                const png = await this.#storage.readFile(getHanzi(ihanzi), iimg + 1, true);
                await png.decompressAndDecodePixels(testPixelBuffer, iimg*npixels);
            }
            for (let [iconfig, config] of this.#trainingConfigs.entries()) {
                const batch =  np.array(testPixelBuffer, { dtype: np.float16, shape: [N_TEST_IMAGES, npixels]});
                const labels = np.full([N_TEST_IMAGES], ihanzi, { dtype: np.uint32 });
                const prediction = config.neuralNet.predict(batch);
                const loss = await prediction.loss(labels.ref).jsAsync();
                const accuracy = await prediction.accuracy(labels);
                prediction.destructor();
                accuracies[iconfig] = (accuracies[iconfig]*ihanzi + accuracy)/(ihanzi + 1);
                losses[iconfig] = (losses[iconfig]*ihanzi + loss)/(ihanzi + 1);
                yield { iconfig, accuracy: accuracies[iconfig], loss: losses[iconfig] };
            }
        }
        
    }
}