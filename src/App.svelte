<script>
  import { onMount } from "svelte";
  import { loadMnist } from "./lib/Mnist";
  import { MNISTDualLayerNN, NeuralNetworkManager } from "./lib/NeuralNetwork";
  import { defaultDevice, init as initJax } from "@jax-js/jax";

  async function init() {
    const devices = await initJax();
    if (devices.includes("webgpu")) defaultDevice("webgpu");
    else if (devices.includes("webgl")) defaultDevice("webgl");
    else if (devices.includes("wasm")) defaultDevice("wasm");
    else defaultDevice("cpu");

    const mnistData = await loadMnist();
    
    const nnManager = new NeuralNetworkManager();
    const mnistNN = new MNISTDualLayerNN();
    nnManager.addModel(mnistNN);
    nnManager.enableModel(0);

    await nnManager.initRandom();
    await nnManager.train(mnistData.train.images, mnistData.train.labels, 1000);
    const { 0: mlpTestResults } = await nnManager.test(mnistData.test.images, mnistData.test.labels);

    console.log(mlpTestResults);

    nnManager.destructor();
  }
  onMount(init);
</script>

<section id="center">
  <div>
    <h1>Hello Neural Network</h1>
  </div>
</section>
