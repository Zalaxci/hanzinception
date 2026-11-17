<script lang="ts">
  import { lax, numpy as np } from "@jax-js/jax";
  import { getHanzi } from "../models/HanziStorage";
  import { onDestroy, onMount } from "svelte";
  import pThrottle from "p-throttle";
  import type { NeuralNetworkModel } from "../models/NeuralNetwork";

  interface Props {
    model: NeuralNetworkModel,
  }
  const { model }: Props = $props();

  let hasDrawn = $state(false);
  let isDrawing = $state(false);
  let lastPos = [0, 0];

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  const lineWidth = 8;

  const pixels = new Float32Array(62500);
  const pixelConvolution = np.array([
    [0.7, 0.7, 0.7, 0.7, 0.7],
    [0.7, 0.9, 0.9, 0.9, 0.7],
    [0.7, 0.9, 1.0, 0.9, 0.7],
    [0.7, 0.9, 0.9, 0.9, 0.7],
    [0.7, 0.7, 0.7, 0.7, 0.7],
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
    if (!ctx || !hasDrawn) return;
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
    if (!ctx) return;
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
    if (!ctx || !isDrawing) return;
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
    if (!ctx) return;
    event.stopPropagation();
    hasDrawn = false;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictedClass = -1;
  }

  onMount(() => {
    ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  })
  onDestroy(() => {
    pixelConvolution.dispose();
  })
</script>
<div style="display: flex; flex-direction: column; border-radius: 1rem;">
  <h3>Draw a hanzi!</h3>
  <div>
    <canvas
      id="hanzi-canvas"
      width="250"
      height="250"
      onpointerdown={drawStart}
      onpointermove={drawMove}
      onpointerleave={drawEnd}
      onpointerup={drawEnd}
      bind:this={canvas}
    ></canvas>
  </div>
  {#if hasDrawn}
      <button onclick={handleClearClicked}>Clear</button>
  {/if}
  {#if predictedClass !== -1}
    <p>Predicted hanzi: {getHanzi(predictedClass)}</p>
    <p>Probability: {predictedProbability}</p>
  {/if}
</div>