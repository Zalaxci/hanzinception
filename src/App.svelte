<script>
  import ImageDownloadView from "./components/ImageDownloadView.svelte";
  import NeuralNetworkTraining from "./components/NeuralNetworkTraining.svelte";

  let imagesAreDownloaded = $state(false);
</script>

<section id="center">
  {#if imagesAreDownloaded}
    <NeuralNetworkTraining/>
  {:else}
    <ImageDownloadView notifyImagesDownloaded={() => imagesAreDownloaded=true}/>
  {/if}
</section>
