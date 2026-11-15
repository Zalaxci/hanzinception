<script lang="ts">
  import { onMount } from "svelte";
  import { createHanziStorageModel, getTotalImageCount, type HanziStorageModel } from "../models/HanziStorage";

  interface Props {
    notifyImagesDownloaded(hanziStorage: HanziStorageModel): void
  }
  const { notifyImagesDownloaded }: Props = $props();

  const ntotalImages = getTotalImageCount();
  let ndownloadedImages = $state(0);
  let errors: String[] = $state([]);

  onMount(async function() {
    const hanziStorage = await createHanziStorageModel();
    hanziStorage.downloadAll(
      (n: number) => {
        ndownloadedImages += n; // n files downloaded
        if (ndownloadedImages === ntotalImages) notifyImagesDownloaded(hanziStorage); // if all downloaded return hanzi storage
      },
      (err: any) => errors.push(new String(err)),
    );
  });
</script>

<section class="main-column">
    <h2>Loading...</h2>
    {#each errors as err}
      <p style="margin: 0.5rem; background-color: yellow; border-radius: 0.5rem; color: lightgrey">{err}</p>
    {/each}
    <h3>{ndownloadedImages} out of {ntotalImages} downloaded</h3>
</section>