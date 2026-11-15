<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { defaultDevice, init as initJax, numpy as np } from "@jax-js/jax";
  import pThrottle from "p-throttle";

  import { HanziRecognizerCNN } from "../models/HanziRecognizer";
  import { getHanzi, getTotalHanziCount, type HanziStorageModel } from "../models/HanziStorage";
  import { HanziRecognitionTrainer } from "../controllers/HanziRecognitionTrainer";

  interface Props {
    hanziStorage: HanziStorageModel
  }
  const { hanziStorage }: Props = $props();

  let trainer: HanziRecognitionTrainer | null = null;
  let trainingLosses: number[] = $state(Array(2).fill(0));
  let testingLosses: number[] = $state(Array(2).fill(0));
  let testingAccuracies: number[] = $state(Array(2).fill(0));

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
    trainer = new HanziRecognitionTrainer(hanziStorage, [
      new HanziRecognizerCNN(getTotalHanziCount()),
    ], 100);
    await trainer.initRandom();
    await trainer.trainAll(20, 0.001, (i: number, loss: number) => trainingLosses[i] = loss);
    await trainer.testAll((i: number, accuracy: number, loss: number) => {
      testingAccuracies[i] = accuracy
      testingLosses[i] = loss
    })
  });
  onDestroy(() => {
    // Clear memory!
    if (trainer !== null)
      trainer.destructor();
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
    // if (!model) return;
    // const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // // First, construct a 2500-dimensional vector from the image data.
    // for (let i = 0; i < 250; i++)
    //   for (let j = 0; j < 250; j++)
    //     pixels[250*i + j] = imgData.data[(250*i + j)*4] / 255;
    // const pixelArray = np.array(pixels).reshape([1, 1, 250, 250]);
    // const shrankArray = lax.convGeneralDilated(pixelArray, pixelConvolution.ref, [5, 5], "VALID");
    // const results = model.predict(shrankArray);
    // predictedClass = (await results.mostLikelyOutputs().data())[0];
    // predictedProbability = (await results.realizeDataAndDestroy())[predictedClass];
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
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>CNN w/ pooling</th>
            <th>Pure CNN (less channels)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>Training loss</th>
            {#each trainingLosses as loss }
              <td>{loss}</td>
            {/each}
          </tr>
          <tr>
            <th>Testing loss</th>
            {#each testingLosses as loss }
              <td>{loss}</td>
            {/each}
          </tr>
          <tr>
            <th>Testing accuracy</th>
            {#each testingAccuracies as accuracy }
              <td>{accuracy}</td>
            {/each}
          </tr>
        </tbody>
      </table>
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
        <p>Predicted hanzi: {getHanzi(predictedClass)}</p>
        <p>Probability: {predictedProbability}</p>
      {/if}
    </div>
  </div>
</section>