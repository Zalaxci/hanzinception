import { lax, nn, numpy as np } from "@jax-js/jax";
import { NeuralNetworkModel } from "./NeuralNetwork";

export class HanziRecognizerMLP extends NeuralNetworkModel {
    constructor(nhanzi: number) {
        super(
            nhanzi,
            // Weight shapes; odd index numbers correspond to matrices we multiply by, even index numbers to offset vectors for each set of neurons
            [[2500, 500], [500], [500, 400], [400], [400, 300], [300], [300, 200], [200], [200, 200], [200], [200, nhanzi], [nhanzi]],
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
    constructor(nhanzi: number) {
        super(
            nhanzi,
            [
                [16, 1, 3, 3],
                [16, 1, 1],
                [64, 16, 3, 3],
                [64, 1, 1],
                [128, 64, 3, 3],
                [128, 1, 1],
                [256, 128, 3, 3],
                [256, 1, 1],
                [256, nhanzi],
                [nhanzi],
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
export class HanziRecognizerPureCNN extends NeuralNetworkModel {
    constructor(nhanzi: number) {
        super(
            nhanzi,
            [
                [16, 1, 3, 3],
                [16, 1, 1],
                [32, 16, 3, 3],
                [32, 1, 1],
                [64, 32, 3, 3],
                [64, 1, 1],
                [128, 64, 3, 3],
                [128, 1, 1],
                [256, 128, 3, 3],
                [256, 1, 1],
                [nhanzi, 256, 1, 1],
                [nhanzi, 1, 1],
            ],
            (w: np.Array[], x: np.Array) => {
                x = x.reshape([-1, 1, 50, 50]); // x is Nx1x50x50 (50x50 pixels, 1 channel)
                const conv1 = lax.convGeneralDilated(x, w[0], [1, 1], "VALID").add(w[1]); // 3x3 convolution with stride 1 => shape Nx16x48x48 (48x48 pixels, 16 channels)
                const a1 = nn.relu(conv1);
                const conv2 = lax.convGeneralDilated(a1, w[2], [2, 2], "VALID").add(w[3]); // 3x3 convolution with stride 2 => shape Nx32x23x23 (23x23 pixels, 32 channels)
                const a2 = nn.relu(conv2);
                const conv3 = lax.convGeneralDilated(a2, w[4], [1, 1], "VALID").add(w[5]); // 3x3 convolution with stride 1 => shape Nx64x21x21 (21x21 pixels, 64 channels)
                const a3 = nn.relu(conv3);
                const conv4 = lax.convGeneralDilated(a3, w[6], [2, 2], "VALID").add(w[7]); // 3x3 convolution with stride 2 => shape Nx128x10x10 (10x10 pixels, 128 channels)
                const a4 = nn.relu(conv4);
                const conv5 = lax.convGeneralDilated(a4, w[8], [2, 2], "VALID").add(w[9]); // 3x3 convolution with stride 2 => shape Nx256x4x4 (4x4 pixels, 256 channels)
                const a5 = nn.relu(conv5);
                const conv6 = lax.convGeneralDilated(a5, w[10], [2, 2], "VALID").add(w[11]); // 1x1 convolution => shape NxNHANZIx1x1 (4x4 pixels, 1 channel per hanzi)
                const a6 = nn.relu(conv6);
                const a7 = lax.reduceWindow(a6, np.average, [4, 4], [4, 4]);
                const conv8flat = a7.reshape([-1, nhanzi]); // NxNHANZI
                return nn.logSoftmax(conv8flat);
            }
        )
        this._initialWeightBound[0] = 1 / 3;
        this._initialWeightBound[1] = 1 / (3 * Math.sqrt(16));
        this._initialWeightBound[2] = 1 / (3 * Math.sqrt(32));
        this._initialWeightBound[3] = 1 / (3 * Math.sqrt(64));
        this._initialWeightBound[4] = 1 / (3 * Math.sqrt(128));
        this._initialWeightBound[5] = 1 / (3 * Math.sqrt(256));
    }
}