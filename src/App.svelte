<script lang="ts">
  import { type HanziStorageModel } from "./models/HanziStorage";
  import ImageDownloadView from "./views/ImageDownloadView.svelte";
  import NNTrainingView from "./views/NNTrainingView.svelte";

  let hanziStorage: HanziStorageModel | null = $state(null);
</script>

{#if hanziStorage}
  <NNTrainingView hanziStorage={hanziStorage}/>
{:else}
  <ImageDownloadView notifyImagesDownloaded={storage => hanziStorage = storage}/>
{/if}
