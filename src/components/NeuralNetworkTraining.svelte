<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  import { trainImageIterator, hanziList, testImageIterator, ntrainImages, ntestImages } from "../lib/HanziReader";
  import { HanziRecognizerCNN, NeuralNetworkModel } from "../lib/NeuralNetwork";
  import { defaultDevice, init as initJax, lax, nn, numpy as np, tree } from "@jax-js/jax";
  import pThrottle from "p-throttle";

  let model: NeuralNetworkModel | null = $state(null);
  const results: string[] = $state([]);
  // Training data: This is stored here so it can be cleared afterwards
  const trainBatchData: (np.Array | undefined)[] = new Array(ntrainImages);

  onMount(async function init() {
    ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const devices = await initJax();
    if (devices.includes("webgpu")) defaultDevice("webgpu");
    else if (devices.includes("webgl")) defaultDevice("webgl");
    else if (devices.includes("wasm")) defaultDevice("wasm");
    else defaultDevice("cpu");

    // Create model & initialize weights
    model = new HanziRecognizerCNN();
    await model.initRandom();
    // Train
    const trainImgIterator = trainImageIterator(50, 50, trainBatchData, hanziList.length);
    const labels = np.arange(hanziList.length);
    const trainingIterator = model.train(trainImgIterator, labels.ref, ntrainImages, 20, 0.0008);
    for await (let metric of trainingIterator)
      results.unshift(`*TRAINING* Epoch: ${metric.iepoch}, Batch: ${metric.ibatch}, Loss: ${metric.loss}`);
    // Free memory used in training
    for (let batch of trainBatchData)
      batch?.dispose();
    // Test
    const testImgIterator = testImageIterator(50, 50);
    let totalAccuracy = 0;
    let totalLoss = 0;
    for await (let testingBatch of testImgIterator) {
      const modelOutput = model.predict(testingBatch);
      const accuracy = await modelOutput.accuracy(labels.ref);
      const loss = await modelOutput.loss(labels.ref).jsAsync();
      results.unshift(`*TESTING* Accuracy: ${accuracy}, Loss: ${loss}`);
      totalAccuracy += accuracy;
      totalLoss += loss;
      modelOutput.destructor();
    }
    results.unshift(`*TESTING AVERAGE* Accuracy: ${totalAccuracy/ntestImages}, Loss: ${totalLoss/ntestImages}`);
    labels.dispose();
  });
  onDestroy(() => {
    // Clear memory!
    if (model !== null) {
      model.destructor();
      model = null;
    }
    // Free memory used in training
    for (let arr of trainBatchData)
      if (arr)
        arr.dispose();
    pixelConvolution.dispose();
  });

  // Drawing functionality (TODO: seperate component)
  const lineWidth = 8;

  let hasDrawn = $state(false);
  let isDrawing = $state(false);
  let lastPos = [0, 0];

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;

  const pixels = new Float32Array(62500);
  const pixelConvolution = np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.8, 0.8, 0.8, 0.5],
    [0.5, 0.8, 1.0, 0.8, 0.5],
    [0.5, 0.8, 0.8, 0.8, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5],
  ]).mul(1/25).reshape([1, 1, 5, 5]);

  let predictedClass: number = $state(-1);
  let predictedProbability: number = $state(0);

  function getCanvasCoords(event: PointerEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    return [
      (event.offsetX / rect.width) * canvas.width,
      (event.offsetY / rect.height) * canvas.height,
    ];
  }
  const inferCanvasHanzi = pThrottle({ limit: 0, interval: 30 })(async () => {
    if (!model) return;
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // First, construct a 2500-dimensional vector from the image data.
    for (let i = 0; i < 250; i++)
      for (let j = 0; j < 250; j++)
        pixels[250*i + j] = imgData.data[(250*i + j)*4] / 255;
    const pixelArray = np.array(pixels).reshape([1, 1, 250, 250]);
    const shrankArray = lax.convGeneralDilated(pixelArray, pixelConvolution.ref, [5, 5], "VALID");
    const results = model.predict(shrankArray);
    predictedClass = (await results.mostLikelyOutputs().data())[0];
    predictedProbability = (await results.realizeDataAndDestroy())[predictedClass];
  })
  function drawStart(event: PointerEvent) {
    event.preventDefault();
    const [x, y] = getCanvasCoords(event);
    isDrawing = true;
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.ellipse(x, y, lineWidth / 2, lineWidth / 2, 0, 0, Math.PI * 2);
    ctx.fill();
    lastPos = [x, y];
    hasDrawn = true;
    inferCanvasHanzi();
  }
  function drawMove(event: PointerEvent) {
    if (!isDrawing) return;
    event.preventDefault();
    const [x, y] = getCanvasCoords(event);
    ctx.beginPath();
    ctx.moveTo(lastPos[0], lastPos[1]);
    ctx.lineTo(x, y);
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();
    lastPos = [x, y];
    inferCanvasHanzi();
  }
  function drawEnd() {
    isDrawing = false;
  }
  function handleClearClicked(event: MouseEvent) {
    event.stopPropagation();
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    predictedClass = -1;
  }
</script>

<section class="main-column">
  <h2>Neural network trainer</h2>
  <div class="row">
    <div style="max-height: 60vh; display: flex; flex-direction: column; overflow-y: scroll;">
      {#each results as resultStr}
        <p>{resultStr}</p>
      {/each}
    </div>
    <div style="display: flex; flex-direction: column; border-radius: 1rem;">
      <h3>Demo!</h3>
      <div style="position: relative; border: solid .4rem indigo; border-radius: .5rem;">
        <canvas
          width="250"
          height="250"
          onpointerdown={drawStart}
          onpointermove={drawMove}
          onpointerleave={drawEnd}
          onpointerup={drawEnd}
          bind:this={canvas}
        ></canvas>
        {#if !hasDrawn}
          <p style="position: absolute; top: 60px; left: 30px; color: navy; font-size: x-large; font-style: italic; transform: rotate(30deg);">Draw a hanzi here!</p>
        {/if}
      </div>
      {#if hasDrawn}
          <button onclick={handleClearClicked}>Clear</button>
      {/if}
      {#if predictedClass !== -1}
        <p>Predicted hanzi: {hanziList[predictedClass]}</p>
        <p>Probability: {predictedProbability}</p>
      {/if}
    </div>
  </div>
</section>