import { numpy as np, tree } from "@jax-js/jax";
import { getHanzi, getImageSize, getTotalHanziCount, getTotalImageCount, N_TEST_IMAGES, N_TRAIN_IMAGES, type HanziStorageModel } from "../models/HanziStorage";
import type { NeuralNetworkModel, NNTrainInput } from "../models/NeuralNetwork";

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

export class HanziRecognitionTrainer {
    #batchSize: number
    #nhanzi: number
    #imageSize: number
    #trainPixelBuffer: Float32Array<ArrayBuffer>
    #trainImageData: (np.Array | undefined)[]
    #storage: HanziStorageModel
    #neuralNets: NeuralNetworkModel[]
    constructor(storage: HanziStorageModel, neuralNets: NeuralNetworkModel[], batchSize: number) {
        this.#batchSize = batchSize;
        this.#nhanzi = getTotalHanziCount();
        this.#imageSize = getImageSize();
        // Temporary container for pixel data, capable of fitting batchSize images to avoid race condition when loading many images asynchronously
        this.#trainPixelBuffer = new Float32Array(this.#batchSize*this.#imageSize*this.#imageSize);
        // 2D array with rows equal to hanzi number & columns the images per hanzi
        this.#trainImageData = new Array(this.#nhanzi*N_TRAIN_IMAGES);
        this.#storage = storage;
        this.#neuralNets = neuralNets;
    }
    destructor() {
        tree.dispose(this.#trainImageData);
        for (let nn of this.#neuralNets) nn.destructor();
    }
    async #readTrainImageDataCached(index: number, bufferpos: number): Promise<np.Array> {
        if (this.#trainImageData[index])
            return this.#trainImageData[index].ref;
        const npixels = this.#imageSize*this.#imageSize;
        const ihanzi = index % this.#nhanzi;
        const iimg = (index - ihanzi)/this.#nhanzi;
        const png = await this.#storage.readFile(getHanzi(ihanzi), iimg + 1, false);
        await png.decompressAndDecodePixels(this.#trainPixelBuffer, bufferpos*npixels);
        this.#trainImageData[index] = np.array(this.#trainPixelBuffer.slice(bufferpos*npixels, (bufferpos + 1)*npixels));
        return this.#trainImageData[index].ref;
    }
    async *#createBatchGenerator(nepochs: number): AsyncGenerator<NNTrainInput> {
        const nbatches = Math.ceil(N_TRAIN_IMAGES * this.#nhanzi / this.#batchSize);
        const epochIndices: number[] = new Array(this.#trainImageData.length);
        for (let iepoch = 0; iepoch < nepochs; iepoch++) {
            randomlyOrderNumbersUpToN(epochIndices);
            for (let ibatch = 0; ibatch < nbatches; ibatch++) {
                const batchIndices = epochIndices.slice(ibatch*this.#batchSize, (ibatch + 1)*this.#batchSize);
                const batchImageArrays = await Promise.all(
                    batchIndices.map(this.#readTrainImageDataCached.bind(this))
                );
                const batchLabels = batchIndices.map(i => i % this.#nhanzi);
                yield {
                    labels: np.array(batchLabels, { dtype: np.uint32 }),
                    data: np.stack(batchImageArrays, 0),
                };
            }
        }
    }
    async initRandom() {
        for (let nn of this.#neuralNets)
            await nn.initRandom();
    }
    async trainAll(nepochs: number, learningRate: number, updateTrainLoss: (imodel: number, loss: number) => void) {
        const nbatches = Math.ceil(N_TRAIN_IMAGES * this.#nhanzi / this.#batchSize);
        const batchGenerator = this.#createBatchGenerator(this.#neuralNets.length*nepochs);
        for (let [i, nn] of this.#neuralNets.entries())
            for await (let metric of nn.train(batchGenerator, nbatches, nepochs, learningRate))
                updateTrainLoss(i, metric.loss);
    }
    async testAll(updateTestLoss: (imodel: number, accuracy: number, loss: number) => void) {
        const npixels = this.#imageSize*this.#imageSize;
        const testImageData: np.Array[] = new Array(this.#nhanzi);
        const testPixelBuffer = new Float32Array(N_TEST_IMAGES*npixels);
        for (let ihanzi = 0; ihanzi < this.#nhanzi; ihanzi++) {
            for (let iimg = 0; iimg < N_TEST_IMAGES; iimg++) {
                const png = await this.#storage.readFile(getHanzi(ihanzi), iimg + 1, true);
                await png.decompressAndDecodePixels(testPixelBuffer, iimg*npixels);
            }
            testImageData[ihanzi] = np.array(testPixelBuffer, { shape: [N_TEST_IMAGES, npixels]});
        }
        for (let [i, nn] of this.#neuralNets.entries()) {
            let avgLoss = 0;
            let avgAccuracy = 0;
            for (let [ibatch, batch] of testImageData.entries()) {
                const labels = np.full([N_TEST_IMAGES], ibatch);
                const prediction = nn.predict(batch);
                const loss = await prediction.loss(labels).jsAsync();
                const accuracy = await prediction.accuracy(labels);
                prediction.destructor();
                avgLoss = (avgLoss*ibatch + loss)/(ibatch + 1);
                avgAccuracy = (avgAccuracy*ibatch + accuracy)/(ibatch + 1);
                updateTestLoss(i, avgAccuracy, avgLoss);
            }
        }
    }
}