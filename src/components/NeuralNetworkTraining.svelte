<script>
  import { onMount } from "svelte";

  import { hanziImageIterator, hanziLabelIterator, imageCountPerHanzi } from "../lib/HanziReader";
  import { HanziRecognizer3LayerMLP, NeuralNetworkManager } from "../lib/NeuralNetwork";
  import { defaultDevice, init as initJax } from "@jax-js/jax";
  import { OPFSFacade } from "opfs-worker";

  /** @type {string[]} */
  const trainingResults = $state([]);

  onMount(async function init() {
    const devices = await initJax();
    if (devices.includes("webgpu")) defaultDevice("webgpu");
    else if (devices.includes("webgl")) defaultDevice("webgl");
    else if (devices.includes("wasm")) defaultDevice("wasm");
    else defaultDevice("cpu");

    const nnManager = new NeuralNetworkManager();
    const hanziNN = new HanziRecognizer3LayerMLP();
    nnManager.addModel(hanziNN);
    nnManager.enableModel(0);

    await nnManager.initRandom();
    const hanziImages = hanziImageIterator(50, 50, 20, true);
    const hanziLabels = hanziLabelIterator(20);

    const trainingIterator = hanziNN.train(hanziImages, hanziLabels, imageCountPerHanzi, 20, 0.005);
    for await (let metric of trainingIterator)
      trainingResults.unshift(`Epoch: ${metric.iepoch}, Batch: ${metric.ibatch}, Loss: ${metric.loss}`);
    //const { 0: mlpTestResults } = await nnManager.test(mnistData.test.images, mnistData.test.labels);

    //console.log(mlpTestResults);

    nnManager.destructor();
  });
</script>

<div>
  <h2>Neural network trainer</h2>
  <div style="max-height: 60vh; overflow: scroll">
    {#each trainingResults as resultStr}
      <p>{resultStr}</p>
    {/each}
  </div>
</div>