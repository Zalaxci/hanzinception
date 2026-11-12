<script>
  import { hanziList, imageCountPerHanzi, testImageCountPerHanzi } from "../lib/HanziReader";
  import { onMount } from "svelte";
  import { createOPFS } from "opfs-worker";

  /** @type {{ notifyImagesDownloaded: Function }} */
  const { notifyImagesDownloaded } = $props();

  const totalImages = hanziList.length * (imageCountPerHanzi + testImageCountPerHanzi);
  let downloadedImagesCount = $state(0);

  const fs = createOPFS({
    root: '/hanzi',
    hashAlgorithm: false,
  });
  /** @type {Promise<void>[]} */
  let promises = [];

  onMount(async function listDownloaded() {
    if (await fs.exists("ALL_DOWNLOADED")) {
      downloadedImagesCount = totalImages;
      notifyImagesDownloaded();
      return;
    }
    // Download & store training data
    for (let ihanzi = 0; ihanzi < hanziList.length; ihanzi++) {
      for (let iimg = 0; iimg < imageCountPerHanzi; iimg++) {
        const imgFilePath = `/Train/${hanziList[ihanzi]}/${iimg + 1}.png`;
        if (await fs.exists(imgFilePath)) downloadedImagesCount++;
        else promises.push(downloadFile(imgFilePath));
        if (promises.length === 200) {
          await Promise.all(promises);
          promises = [];
        }
      }
    }
    // Download & store test data
    for (let ihanzi = 0; ihanzi < hanziList.length; ihanzi++) {
      for (let iimg = 0; iimg < testImageCountPerHanzi; iimg++) {
        const imgFilePath = `/Test/${hanziList[ihanzi]}/${iimg + 1}.png`;
        if (await fs.exists(imgFilePath)) downloadedImagesCount++;
        else promises.push(downloadFile(imgFilePath));
        if (promises.length === 200) {
          await Promise.all(promises);
          promises = [];
        }
      }
    }
    await Promise.all(promises);
    await fs.writeText("ALL_DOWNLOADED", "");
    notifyImagesDownloaded();
  });
  /**
   * @param imgFilePath {string}
   */
  async function downloadFile(imgFilePath) {
    const resp = await fetch("assets/chinese-handwriting" + imgFilePath);
    const imgData = await resp.bytes();
    await fs.writeFile(imgFilePath, imgData);
    downloadedImagesCount++;
  }
</script>

<section class="main-column">
    <h2>Loading...</h2>
    <h3>{downloadedImagesCount} out of {totalImages} downloaded</h3>
</section>