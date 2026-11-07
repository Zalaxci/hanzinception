import { numpy as np } from "@jax-js/jax";
import { cachedFetch } from "@jax-js/loaders";

const MNIST_URLS = [
    "https://cdn.jsdelivr.net/gh/fgnt/mnist/train-images-idx3-ubyte.gz", // Training images
    "https://cdn.jsdelivr.net/gh/fgnt/mnist/train-labels-idx1-ubyte.gz", // Training labels
    "https://cdn.jsdelivr.net/gh/fgnt/mnist/t10k-images-idx3-ubyte.gz", // Testing images
    "https://cdn.jsdelivr.net/gh/fgnt/mnist/t10k-labels-idx1-ubyte.gz", // Testing labels
]
interface IdxResults {
    shape: number[]
    data: Int32Array<ArrayBuffer> | Float32Array<ArrayBuffer>
}
export async function loadGzipIdx(url: string, useFloatArray: boolean): Promise<IdxResults> {
    const bytes = await cachedFetch(url);
    const stream = new Blob([bytes])
        .stream()
        .pipeThrough(new DecompressionStream("gzip"));
    const uncompressedData = await new Response(stream).arrayBuffer();
    let currView = new DataView(uncompressedData);
    const dataType = currView.getUint8(2);
    const rank = currView.getUint8(3);
    const shape: number[] = new Array(rank);
    let len = 1;
    for (let i = 0; i < rank; i++) {
        shape[i] = currView.getUint32(4 + 4*i, false);
        len *= shape[i];
    }
    currView = new DataView(uncompressedData, 4 + 4*rank);
    const data = useFloatArray? new Float32Array(len) : new Int32Array(len);
    switch (dataType) {
    case 0x08: // unsigned byte
        for (let i = 0; i < len; i++) data[i] = currView.getUint8(i);
        return { shape, data };
    case 0x09: // signed byte
        for (let i = 0; i < len; i++) data[i] = currView.getInt8(i);
        return { shape, data };
    case 0x0b: // short (2 bytes)
        for (let i = 0; i < len; i++) data[i] = currView.getInt16(i * 2, false); // Big-endian
        return { shape, data };
    case 0x0c: // int (4 bytes)
        for (let i = 0; i < len; i++) data[i] = currView.getInt32(i * 4, false); // Big-endian
        return { shape, data };
    case 0x0d: // float (4 bytes)
        for (let i = 0; i < len; i++) data[i] = currView.getFloat32(i * 4, false); // Big-endian
        return { shape, data };
    case 0x0e: // double (8 bytes)
        for (let i = 0; i < len; i++) data[i] = currView.getFloat64(i * 8, false); // Big-endian
        return { shape, data };
    default:
      throw new Error(`Unsupported data type: ${dataType}`);
  }
}
export async function loadMnist() {
    const results = await Promise.all([
        loadGzipIdx(MNIST_URLS[0], true),
        loadGzipIdx(MNIST_URLS[1], false),
        loadGzipIdx(MNIST_URLS[2], true),
        loadGzipIdx(MNIST_URLS[3], false),
    ])
    return {
        train: {
            images: np.array(results[0].data, { shape: [results[0].shape[0], 784], dtype: np.float32 }).mul(1/255),
            labels: np.array(results[1].data, { dtype: np.int32 }),
        },
        test: {
            images: np.array(results[2].data, { shape: [results[2].shape[0], 784], dtype: np.float32 }).mul(1/255),
            labels: np.array(results[3].data, { dtype: np.int32 }),
        },
    };
}