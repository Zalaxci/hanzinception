<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  import { trainImageIterator, hanziLabelIterator, hanziList, imageCountPerHanzi, testImageIterator } from "../lib/HanziReader";
  import { HanziRecognizer3LayerMLP, NeuralNetworkModel } from "../lib/NeuralNetwork";
  import { defaultDevice, init as initJax, nn, numpy as np } from "@jax-js/jax";
  import pThrottle from "p-throttle";

  let model: NeuralNetworkModel | null = $state(null);
  const results: string[] = $state([]);

  onMount(async function init() {
    ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const devices = await initJax();
    if (devices.includes("webgpu")) defaultDevice("webgpu");
    else if (devices.includes("webgl")) defaultDevice("webgl");
    else if (devices.includes("wasm")) defaultDevice("wasm");
    else defaultDevice("cpu");

    model = new HanziRecognizer3LayerMLP();
    await model.initRandom();
    const trainImages = trainImageIterator(50, 50, 10);
    const testImages = testImageIterator(50, 50);
    const hanziLabels = hanziLabelIterator(10);

    const trainingIterator = model.train(trainImages, hanziLabels, imageCountPerHanzi, 10, 0.001);
    for await (let metric of trainingIterator)
      results.unshift(`*TRAINING* Epoch: ${metric.iepoch}, Batch: ${metric.ibatch}, Loss: ${metric.loss}`);
    for await (let testingBatch of testImages) {
      const modelOutput = model.predict(testingBatch);
      const labels = (await hanziLabels.next()).value as np.Array;
      results.unshift(`*TESTING* Accuracy: ${await modelOutput.accuracy(labels.ref)}, Loss: ${await modelOutput.loss(labels).jsAsync()}`);
      modelOutput.destructor();
    }
  });
  onDestroy(() => {
    // Clear memory!
    if (model !== null) {
      model.destructor();
      model = null;
    }
  })

  // Drawing functionality (TODO: seperate component)
  const lineWidth = 10;

  let hasDrawn = $state(false);
  let isDrawing = $state(false);
  let lastPos = [0, 0];

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;

  let predictedProbabilities: number[] = $state([]);
  //@ts-ignore
  let filteredProbabilities: [string, number][] = $derived(predictedProbabilities.map((p, i) => [hanziList[i], p]).filter(([hanzi, p]) => p > 0.01));

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
    const pixels = new Float32Array(2500);
    for (let i = 0; i < 50; i++) {
      for (let j = 0; j < 50; j++) {
        for (let l = i * 5; l < (i + 1) * 5; l++) {
          for (let k = j * 5; k < (j + 1) * 5; k++) {
            const idx = (l*250 + k)*4;
            const r = imgData.data[idx];
            // Don't verage the RGB values to get a grayscale value, red channel only is the same for monochrome image :D
            // Divide by 25 to average over all pixels
            pixels[i*50 + j] += r/255/25;
          }
        }
      }
    }
    const jaxArray = np.array(pixels).reshape([1, 2500]);
    const results = model.predict(jaxArray);
    predictedProbabilities = await np.exp(results._logits.ref).jsAsync();
    console.log(await results.loss(np.array([[0]], { dtype: np.int32 })).jsAsync());
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
    predictedProbabilities = [];
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
    <div style="display: flex; flex-direction: column; border: solid lightslategrey; border-radius: 1rem;">
      <h3>Demo!</h3>
      {#each filteredProbabilities as [hanzi, p]}
        <p>{hanzi}: probability {p}</p>
      {/each}
      <div style="position: relative; border: solid darkgrey; border-radius: .5rem;">
        <canvas
          width="250"
          height="250"
          onpointerdown={drawStart}
          onpointermove={drawMove}
          onpointerleave={drawEnd}
          onpointerup={drawEnd}
          bind:this={canvas}
        ></canvas>
        {#if hasDrawn}
          <button onclick={handleClearClicked} style="position: absolute; bottom: 10px; right: 10px;">Clear</button>
        {:else}
          <p style="position: absolute; top: 50px; left: 10px; color: navy; font-style: italic; transform: rotate(30)">Draw a hanzi here!</p>
        {/if}
      </div>
    </div>
  </div>
</section>