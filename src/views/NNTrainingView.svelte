<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { defaultDevice, init as initJax } from "@jax-js/jax";

  import type { NNTrainMetrics } from "../models/NeuralNetwork";
  import { HanziRecognizerInception } from "../models/HanziRecognizer";
  import { getTotalHanziCount, type HanziStorageModel } from "../models/HanziStorage";
  import { HanziRecognitionTrainer } from "../controllers/HanziRecognitionTrainer";
  import HanziDrawingView from "./HanziDrawingView.svelte";

  interface Props {
    hanziStorage: HanziStorageModel
  }
  const { hanziStorage }: Props = $props();

  let stage: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 = $state(0);
  let stageMsg = $derived(getStateMsg(stage));
  let inception: HanziRecognizerInception | null = $state(null);
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

    inception = new HanziRecognizerInception(getTotalHanziCount())
    trainer = new HanziRecognitionTrainer(hanziStorage, [
      inception,
    ], 80);
    stage = 2; // Succesfully created inception model
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
        return "Creating model...";
      case 2:
        return "Model created!";
      case 3:
        return "Randomizing weights...";
      case 4:
        return "Training...";
      case 5:
        return "Model trained!"
      case 6:
        return "Loading test data...";
      case 7:
        return "Testing...";
      case 8:
        return "Finished testing!";
      default:
        return "";
    }
  }
  async function loadStored() {
    if (!trainer || !inception) return;
    await inception.loadWeights();
    stage = 5 // Model trained!
  }
  async function trainModel() {
    if (!trainer || !inception) return;
    stage = 3; // Randomizing weights...
    await trainer.initRandom();
    stage = 4; // Weights initialized, training...
    for await (let m of trainer.trainAll(5, 0.001, 0.02))
      trainingMetrics = m.metrics;
    stage = 5; // Model trained!
  }
  async function testModel() {
    if (!trainer || !inception) return;
    stage = 6; // Loading test data...
    await trainer.testAll((i: number, accuracy: number, loss: number) => {
      if (stage === 5) stage = 6; //Testing!
      testingAccuracy = accuracy;
      testingLoss = loss;
    });
    // Finished!
    stage = 7;
  }
</script>

<h2>{stageMsg}</h2>
{#if stage == 2}
  <div>
    <button onclick={loadStored}>Load stored weights!</button>
    <button onclick={trainModel}>Train model!</button>
  </div>
{:else if stage == 5}
  <div>
    <button onclick={testModel}>Test model!</button>
  </div>
{/if}
{#if inception && stage >= 4}
  <HanziDrawingView model={inception}/>
{/if}
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