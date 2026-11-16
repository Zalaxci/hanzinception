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
// GoogleNet-style inception module
// Since this is a simplified implementation of inception with the aim of working in a budget GPU, we have 3 routes in total
// route 1: 3x3 max pooling with stride 1, followed by 1x1 convolution
// route 2: 1x1 convolution (channel reduction) followed by 3x3 convolution
// route 3: 1x1 convolution (channel reduction) followed by a padded 5x5 convolution
function getInceptionShape(inputChannels: number, bottleneckChannels: number[], conv3x3Channels: number, conv5x5Channels: number): number[][] {
    return [
        // "Bottleneck" 1x1 convolutions act like matrix transformations in channel space, reducing the channel number by projecting each pixel in a lower dimensional space
        [bottleneckChannels[0], inputChannels, 1, 1],
        [bottleneckChannels[1], inputChannels, 1, 1],
        [bottleneckChannels[2], inputChannels, 1, 1],
        // For "route 1", an offset is added after max pooling
        [bottleneckChannels[0], 1, 1],
        // Finally, for route 2 and 3 both the convolution itself and the final offset need be stored as weights
        [conv3x3Channels, bottleneckChannels[1], 3, 3],
        [conv3x3Channels, 1, 1],
        [conv5x5Channels, bottleneckChannels[2], 5, 5],
        [conv5x5Channels, 1, 1],
    ]
}
function setInceptionStartingWeightBounds(bounds: Record<number, number>, offset: number, inputChannels: number, bottleneckChannels: number[]) {
    // For both the "bottleneck" convolutions and route 1 offset, the range of starting values depends only on input channel number
    bounds[offset] = 1/Math.sqrt(inputChannels);
    bounds[offset + 1] = 1/Math.sqrt(inputChannels);
    // For the 3x3 convolution and route 2's respective offset it depends on the channels of the input to 3x3 convolution, aka the output of the second "bottleneck" convolution
    bounds[offset + 2] = 1/(3*Math.sqrt(bottleneckChannels[1]));
    // Similarly for the 5x5 convolution
    bounds[offset + 3] = 1/(5*Math.sqrt(bottleneckChannels[2]));
}
function calculateInception(
    x: np.Array,
    bottleneckConvolutions: [np.Array, np.Array, np.Array],
    paddingMatrices: [np.Array, np.Array, np.Array],
    convolution3x3: np.Array,
    convolution5x5: np.Array,
): np.Array {
    const route1a = lax.reduceWindow(x.ref, np.max, [3, 3], [1, 1]);
    const route1b = lax.convGeneralDilated(route1a, bottleneckConvolutions[0], [1, 1], "VALID");
    const route1c = route1b.add(paddingMatrices[0]);
    const route2a = lax.convGeneralDilated(x.ref, bottleneckConvolutions[1], [1, 1], "VALID");
    const route2b = lax.convGeneralDilated(route2a, convolution3x3, [1, 1], "VALID");
    const route2c = route2b.add(paddingMatrices[1]);
    const route3a = lax.convGeneralDilated(x, bottleneckConvolutions[2], [1, 1], "VALID");
    const route3b = lax.convGeneralDilated(route3a, convolution5x5, [1, 1], [[1, 1], [1, 1]]);
    const route3c = route3b.add(paddingMatrices[2]);
    return nn.relu(np.hstack([route1c, route2c, route3c]));
}
export class HanziRecognizerInception extends NeuralNetworkModel {
    constructor(nhanzi: number) {
        super(
            nhanzi,
            [
                [64, 1, 5, 5],
                [64, 1, 1],
                ...getInceptionShape(64, [64, 64, 16], 128, 32),
                ...getInceptionShape(224, [64, 64, 16], 128, 32),
                [100, 224, 1, 1],
                [100, 1, 1],
                [400, nhanzi],
                [nhanzi],
            ],
            (w: np.Array[], x: np.Array): np.Array => {
                x = x.reshape([-1, 1, 50, 50]);
                const conv1 = lax.convGeneralDilated(x, w[0], [1, 1], "VALID").add(w[1]);
                const a1 = nn.relu(conv1); // 64x46x46
                const a2 = lax.reduceWindow(a1, np.max, [2, 2], [2, 2]); // 64x23x23
                const a3 = calculateInception(
                    a2,
                    [w[2], w[3], w[4]],
                    [w[5], w[7], w[9]],
                    w[6],
                    w[8],
                ); // 224x21x21
                const a4 = lax.reduceWindow(a3, np.max, [3, 3], [2, 2]); // 224x10x10
                const a5 = calculateInception(
                    a4,
                    [w[10], w[11], w[12]],
                    [w[13], w[15], w[17]],
                    w[14],
                    w[16],
                ); // 224x8x8
                const conv6 = lax.convGeneralDilated(a5, w[18], [1, 1], "VALID").add(w[19]);
                const a6 = nn.relu(conv6); // 100x8x8
                const a7 = lax.reduceWindow(a6, (x: np.Array) => np.sum(x).mul(1/x.size), [4, 4], [4, 4]); // 100x2x2
                const z8 = np.dot(a7.reshape([-1, 400]), w[20]).add(w[21]);
                return nn.logSoftmax(z8);
            }
        );
        this._initialWeightBound[0] = 1/5;
        setInceptionStartingWeightBounds(this._initialWeightBound, 1, 64, [64, 64, 16]);
        setInceptionStartingWeightBounds(this._initialWeightBound, 5, 224, [64, 64, 16]);
        this._initialWeightBound[9] = 1/Math.sqrt(224);
    }
}