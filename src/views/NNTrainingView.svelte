<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { defaultDevice, init as initJax } from "@jax-js/jax";

  import type { NeuralNetworkModel, NNTrainMetrics } from "../models/NeuralNetwork";
  import { HanziRecognizerCNN, HanziRecognizerInception } from "../models/HanziRecognizer";
  import { getTotalHanziCount, type HanziStorageModel } from "../models/HanziStorage";
  import { HanziRecognitionTrainer } from "../controllers/HanziRecognitionTrainer";
  import HanziDrawingView from "./HanziDrawingView.svelte";
  import ImageDownloadView from "./ImageDownloadView.svelte";

  let hanziStorage: HanziStorageModel | null = $state(null);

  let stage: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 = $state(0);
  let stageMsg = $derived(getStateMsg(stage));
  let hanziRecognizerNN: NeuralNetworkModel | null = $state(null);
  let trainer: HanziRecognitionTrainer | null = $state(null);
  let trainingMetrics: NNTrainMetrics | null = $state(null);
  let testingLoss = $state(0);
  let testingAccuracy = $state(0);

  onMount(async function init() {
    const devices = await initJax();
    if (devices.includes("webgpu")) defaultDevice("webgpu");
    else if (devices.includes("webgl")) defaultDevice("webgl");
    else if (devices.includes("wasm")) defaultDevice("wasm");
    else defaultDevice("cpu");
    stage = 1; // Succesfully initialized Jax & set Jax backend
  });
  onDestroy(async () => {
    // Clear memory!
    if (trainer !== null)
      trainer.destructor();
  });

  function getStateMsg(s: number) {
    switch (s) {
      case 0:
        return "Initializing Jax...";
      case 1:
        return "Jax initialized!";
      case 2:
        return "Creating model...";
      case 3:
        return "Model created!";
      case 4:
        return "Randomizing weights...";
      case 5:
        return "Training...";
      case 6:
        return "Model trained!"
      case 7:
        return "Loading test data...";
      case 8:
        return "Testing...";
      case 9:
        return "Finished testing!";
      default:
        return "";
    }
  }
  async function loadStored() {
    stage = 2; // Creating model...
    hanziRecognizerNN = new HanziRecognizerInception(getTotalHanziCount());
    stage = 3; // Succesfully created hanzi inception model
    await hanziRecognizerNN.loadWeights();
    stage = 6 // Model trained!
  }
  async function trainModel() {
    if (!trainer) return;
    stage = 2; // Creating model...
    hanziRecognizerNN = new HanziRecognizerCNN(getTotalHanziCount());
    trainer.addNN(hanziRecognizerNN);
    stage = 4; // Randomizing weights...
    await hanziRecognizerNN.initRandom();
    stage = 5; // Weights initialized, training...
    for await (let m of trainer.trainAll(5, 0.001, 0.02))
      trainingMetrics = m.metrics;
    stage = 6; // Model trained!
  }
  async function testModel() {
    if (!trainer || !hanziRecognizerNN) return;
    stage = 7; // Loading test data...
    await trainer.testAll((i: number, accuracy: number, loss: number) => {
      if (stage === 7) stage = 8; //Testing!
      testingAccuracy = accuracy;
      testingLoss = loss;
    });
    // Finished!
    stage = 9;
  }
  function notifyImagesDownloaded(storage: HanziStorageModel) {
    hanziStorage = storage;
    trainer = new HanziRecognitionTrainer(hanziStorage, 80);
    if (hanziRecognizerNN) trainer.addNN(hanziRecognizerNN);
  }
</script>

<section class="main-column">
  {#if hanziStorage}
    <h2>{stageMsg}</h2>
  {:else}
    <ImageDownloadView notifyImagesDownloaded={notifyImagesDownloaded}/>
  {/if}
  <div>
    {#if stage === 1}<button onclick={loadStored}>Load stored weights!</button>{/if}
    {#if stage === 1 && trainer}<button onclick={trainModel}>Train model!</button>{/if}
    {#if stage === 6 && trainer}<button onclick={testModel}>Test model!</button>{/if}
  </div>
  {#if hanziRecognizerNN}
    <HanziDrawingView model={hanziRecognizerNN}/>
  {/if}
  {#if trainer}
    <div class="row">
      {#if trainingMetrics}
        <div id="training-metrics" class="section">
          <h3>Training metrics:</h3>
          <table>
            <tbody>
              <tr>
                <th>Loss</th>
                <td>{trainingMetrics.loss}</td>
              </tr>
              <tr>
                <th>Epoch</th>
                <td>{trainingMetrics.iepoch}</td>
              </tr>
              <tr>
                <th>Batch</th>
                <td>{trainingMetrics.ibatch}</td>
              </tr>
            </tbody>
          </table>
        </div>
      {/if}
      {#if stage >= 6}
        <div id="training-metrics" class="section">
          <h3>Testing metrics:</h3>
          <table>
            <tbody>
              <tr>
                <th>Loss</th>
                <td>{testingLoss}</td>
              </tr>
              <tr>
                <th>Accuracy</th>
                <td>{testingAccuracy}</td>
              </tr>
            </tbody>
          </table>
        </div>
      {/if}
    </div>
  {/if}
</section>