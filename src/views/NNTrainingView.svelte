<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { defaultDevice, init as initJax } from "@jax-js/jax";

  import type { NNTrainMetrics } from "../models/NeuralNetwork";
  import { HanziRecognizerCNN, HanziRecognizerInception } from "../models/HanziRecognizer";
  import { getTotalHanziCount, type HanziStorageModel } from "../models/HanziStorage";
  import { HanziRecognitionTrainer, type NNTrainingConfiguration } from "../controllers/HanziRecognitionTrainer";
  import HanziDrawingView from "./HanziDrawingView.svelte";
  import ImageDownloadView from "./ImageDownloadView.svelte";

  let hanziStorage: HanziStorageModel | null = $state(null);

  let stage: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 = $state(0);
  let stageMsg = $derived(getStateMsg(stage));
  let trainingConfigs: NNTrainingConfiguration[] = $state([]);
  let selectedConfig = $state(0);
  let trainer: HanziRecognitionTrainer | null = $state(null);
  let trainingMetrics: NNTrainMetrics[] = $state([]);
  let testingMetrics: { loss: number, accuracy: number }[] = $state([]);

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
    const inception = new HanziRecognizerInception(getTotalHanziCount());
    trainingConfigs = [
      {
        modelName: "Inception",
        neuralNet: inception,
        batchSize: 80,
        learningRate: 0.001,
      },
    ];
    if (trainer) trainer.addConfigurations(...trainingConfigs);
    stage = 3; // Succesfully created hanzi inception model
    await inception.loadWeights();
    stage = 6 // Model trained!
  }
  async function trainModels() {
    if (!trainer) return;
    stage = 2; // Creating model...
    trainingConfigs = [
      {
        modelName: "Simple CNN",
        neuralNet: new HanziRecognizerCNN(getTotalHanziCount()),
        batchSize: 60,
        learningRate: 0.001,
      },
      {
        modelName: "Simple CNN",
        neuralNet: new HanziRecognizerCNN(getTotalHanziCount()),
        batchSize: 80,
        learningRate: 0.0008,
      },
      {
        modelName: "Simple CNN",
        neuralNet: new HanziRecognizerCNN(getTotalHanziCount()),
        batchSize: 80,
        learningRate: 0.001,
      },
    ];
    trainer.addConfigurations(...trainingConfigs);
    stage = 4; // Randomizing weights...
    await trainer.initRandom();
    stage = 5; // Weights initialized, training...
    for await (let { iconfig, metrics } of trainer.trainAll(8, 0.02))
      trainingMetrics[iconfig] = metrics;
    stage = 6; // Model trained!
  }
  async function testModels() {
    if (!trainer || trainingConfigs.length === 0) return;
    stage = 7; // Loading test data...
    for await (let { iconfig, accuracy, loss } of trainer.testAll())
      testingMetrics[iconfig] = { accuracy, loss };
    // Finished!
    stage = 9;
  }
  function notifyImagesDownloaded(storage: HanziStorageModel) {
    hanziStorage = storage;
    trainer = new HanziRecognitionTrainer(hanziStorage);
    trainer.addConfigurations(...trainingConfigs);
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
    {#if stage === 1 && trainer}<button onclick={trainModels}>Train model!</button>{/if}
    {#if stage === 6 && trainer}<button onclick={testModels}>Test model!</button>{/if}
  </div>
  {#if trainingConfigs[selectedConfig]}
    <HanziDrawingView model={trainingConfigs[selectedConfig].neuralNet}/>
  {/if}
  {#if stage >= 3}
    <div id="configurations-view">
      {#each trainingConfigs as config, i}
        <span>Config {i + 1}: Model {config.modelName} with batch size {config.batchSize} & learning rate {config.learningRate}</span>
        <button onclick={() => selectedConfig = i}>Select config {i + 1}!</button>
        <br>
      {/each}
    </div>
    <div class="row">
      {#if trainingMetrics.length > 0}
        <div id="training-metrics" class="section">
          <h3>Training metrics</h3>
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Loss</th>
                <th>Epoch</th>
                <th>Batch</th>
              </tr>
            </thead>
            <tbody>
              {#each trainingMetrics as metric, i }
                <tr>
                  <th>Configuration {i + 1}: </th>
                  <td>{metric.loss}</td>
                  <td>{metric.iepoch}</td>
                  <td>{metric.ibatch}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </div>
    <div class="row">
      {#if testingMetrics.length > 0}
        <div id="training-metrics" class="section">
          <h3>Testing metrics</h3>
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Loss</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {#each testingMetrics as metric, i }
                <tr>
                  <th>Configuration {i + 1}: </th>
                  <td>{metric.loss}</td>
                  <td>{metric.accuracy}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </div>
  {/if}
</section>