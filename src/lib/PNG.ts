import { cachedFetch } from '@jax-js/loaders';

export default class PNGParser {
  // Width & height
  width: number
  height: number
  // Originally a buffer containing all bytes of the image file, this is overwritten to contain only bytes which constitute compressed image data
  imgData: Uint8Array
  filePos: number // position in the entire image file
  idatLength: number // length of "IDAT" section (actual image data)
  // Properties needed to decode pixels
  ncolors: number
  nbytesPerPixel: number
  compressionMethod: number = 0
  filterMethod: number = 0
  interlaceMethod: number = 0
  // Decompressed data
  decompressedData: Uint8Array | null = null
  dpos: number = 0 // position in decompressed data

  // Parse image
  constructor(w: number, h: number, fileData: Uint8Array) {
    this.width = w;
    this.height = h;
    // Originally, image data array starts out containing the entirity of the input file, or "file data"
    this.imgData = fileData;
    this.filePos = 8; // Skip the default header
    this.idatLength = 0;
    // By default, number of colors is 1, bits per color is 3 & palette is null
    this.ncolors = 1;
    this.nbytesPerPixel = 1;
    while (true) {
      // Read size of section data & increment position
      const chunkSize = this.readUInt32();
      // Read current section identifier & increment position
      let section = "";
      for (let i = 0; i < 4; i++)
        section += String.fromCharCode(fileData[this.filePos++]);
      switch (section) {
        case 'IHDR':
          this.filePos += 8; // Ignore height & width (uint32, 2x4 = 8 bytes in size)
          this.nbytesPerPixel = fileData[this.filePos++] / 8;
          const colorType = fileData[this.filePos++];
          if (colorType === 2 || colorType === 6) this.ncolors = 3; // Is not monochrome => therefore number of colors is 3
          if (colorType === 4 || colorType === 6) this.ncolors++; // Has alpha channel for transparency
          this.compressionMethod = fileData[this.filePos++];
          this.filterMethod = fileData[this.filePos++];
          this.interlaceMethod = fileData[this.filePos++];
          break;

        case 'IDAT':
          this.idatLength = chunkSize;
          return;

        case 'IEND':
          return;

        default:
          // Unknown (or unimportant) section, skip it
          this.filePos += chunkSize;
      }

      this.filePos += 4; // Skip the CRC integrity check

      if (this.filePos > this.imgData.length)
        throw new Error('Incomplete or corrupt PNG file');
    }
  }
  // Fetch image from url
  static async fromURL(w: number, h: number, imgUrl: string | URL) {
    const resp = await fetch(imgUrl);
    return new PNGParser(w, h, await resp.bytes());
  }

  read(nbytes: number) {
    const result = new Uint8Array(nbytes);
    for (let i = 0; i < nbytes; i++)
      result[i] = this.imgData[this.filePos++];
    return result;
  }
  readUInt32() {
    const b1 = this.imgData[this.filePos++] << 24;
    const b2 = this.imgData[this.filePos++] << 16;
    const b3 = this.imgData[this.filePos++] << 8;
    const b4 = this.imgData[this.filePos++];
    return b1 | b2 | b3 | b4;
  }
  readUInt16() {
    const b1 = this.imgData[this.filePos++] << 8;
    const b2 = this.imgData[this.filePos++];
    return b1 | b2;
  }
  async decompress(): Promise<void> {
    const stream = new Blob([ this.imgData.slice(this.filePos, this.filePos + this.idatLength) ])
                    .stream()
                    .pipeThrough(new DecompressionStream("deflate"));
    this.decompressedData = await new Response(stream).bytes();
  }
  decodeSegment(pixels: Float32Array<ArrayBuffer>, offset: number, x0: number, y0: number, dx: number, dy: number, singlePass = false) {
    // Ensure decompressed data is non-null
    if (this.decompressedData === null)
      throw new Error("cannot read decompressed data as it is null - please read an image with 'decompress' before trying to call 'decodeSegment' method");
    const w = Math.ceil((this.width - x0) / dx);
    const h = Math.ceil((this.height - y0) / dy);
    const scanlineLength = this.nbytesPerPixel * w;
    const buffer = singlePass ? pixels : new Float32Array(scanlineLength * h);
    let row = 0;
    let c = offset;
    while (row < h && this.dpos < this.decompressedData.length) {
      let byte: number, col: number, i: number, left: number, upper: number;
      switch (this.decompressedData[this.dpos++]) {
        case 0: // None
          for (i = 0; i < scanlineLength; i++)
            buffer[c++] = this.decompressedData[this.dpos++];
          break;

        case 1: // Sub
          for (i = 0; i < scanlineLength; i++) {
            byte = this.decompressedData[this.dpos++];
            left = i < this.nbytesPerPixel ? 0 : buffer[c - this.nbytesPerPixel];
            buffer[c++] = (byte + left) % 256;
          }
          break;

        case 2: // Up
          for (i = 0; i < scanlineLength; i++) {
            byte = this.decompressedData[this.dpos++];
            col = (i - (i % this.nbytesPerPixel)) / this.nbytesPerPixel;
            upper =
              row &&
              buffer[
                (row - 1) * scanlineLength +
                  col * this.nbytesPerPixel +
                  (i % this.nbytesPerPixel)
              ];
            buffer[c++] = (upper + byte) % 256;
          }
          break;

        case 3: // Average
          for (i = 0; i < scanlineLength; i++) {
            byte = this.decompressedData[this.dpos++];
            col = (i - (i % this.nbytesPerPixel)) / this.nbytesPerPixel;
            left = i < this.nbytesPerPixel ? 0 : buffer[c - this.nbytesPerPixel];
            upper =
              row &&
              buffer[
                (row - 1) * scanlineLength +
                  col * this.nbytesPerPixel +
                  (i % this.nbytesPerPixel)
              ];
            buffer[c++] = (byte + Math.floor((left + upper) / 2)) % 256;
          }
          break;

        case 4: // Paeth
          for (i = 0; i < scanlineLength; i++) {
            let paeth: number, upperLeft: number;
            byte = this.decompressedData[this.dpos++];
            col = (i - (i % this.nbytesPerPixel)) / this.nbytesPerPixel;
            left = i < this.nbytesPerPixel ? 0 : buffer[c - this.nbytesPerPixel];

            if (row === 0) {
              upper = upperLeft = 0;
            } else {
              upper =
                buffer[
                  (row - 1) * scanlineLength +
                    col * this.nbytesPerPixel +
                    (i % this.nbytesPerPixel)
                ];
              upperLeft =
                col &&
                buffer[
                  (row - 1) * scanlineLength +
                    (col - 1) * this.nbytesPerPixel +
                    (i % this.nbytesPerPixel)
                ];
            }

            const p = left + upper - upperLeft;
            const pa = Math.abs(p - left);
            const pb = Math.abs(p - upper);
            const pc = Math.abs(p - upperLeft);

            if (pa <= pb && pa <= pc) {
              paeth = left;
            } else if (pb <= pc) {
              paeth = upper;
            } else {
              paeth = upperLeft;
            }

            buffer[c++] = (byte + paeth) % 256;
          }
          break;

        default:
          throw new Error(`Invalid filter algorithm: ${this.decompressedData[this.dpos - 1]}`);
      }

      if (!singlePass) {
        let pixelsPos = ((y0 + row * dy) * this.width + x0) * this.nbytesPerPixel;
        let bufferPos = row * scanlineLength;
        for (i = 0; i < w; i++) {
          for (let j = 0; j < this.nbytesPerPixel; j++)
            pixels[pixelsPos++] = buffer[bufferPos++];
          pixelsPos += (dx - 1) * this.nbytesPerPixel;
        }
      }

      row++;
    }
  }
  async decompressAndDecodePixels(outputArr: Float32Array<ArrayBuffer>, offset: number = 0) {
    await this.decompress();
    if (this.interlaceMethod === 1) {
      /*
        1 6 4 6 2 6 4 6
        7 7 7 7 7 7 7 7
        5 6 5 6 5 6 5 6
        7 7 7 7 7 7 7 7
        3 6 4 6 3 6 4 6
        7 7 7 7 7 7 7 7
        5 6 5 6 5 6 5 6
        7 7 7 7 7 7 7 7
      */
      this.decodeSegment(outputArr, offset, 0, 0, 8, 8); // 1
      this.decodeSegment(outputArr, offset, 4, 0, 8, 8); // 2
      this.decodeSegment(outputArr, offset, 0, 4, 4, 8); // 3
      this.decodeSegment(outputArr, offset, 2, 0, 4, 4); // 4
      this.decodeSegment(outputArr, offset, 0, 2, 2, 4); // 5
      this.decodeSegment(outputArr, offset, 1, 0, 2, 2); // 6
      this.decodeSegment(outputArr, offset, 0, 1, 1, 2); // 7
    } else {
      this.decodeSegment(outputArr, offset, 0, 0, 1, 1, true);
    }
    const nbytesInImage = this.width * this.height * this.nbytesPerPixel;
    for (let i = 0; i < nbytesInImage; i++) outputArr[offset + i] /= 255;
  }
  // Copy pixels without reverse filtering - this seems to make the neural network converge
  async decompressAndCopyPixels(outputArr: Float32Array<ArrayBuffer>, offset: number = 0) {
    await this.decompress();
    // Ensure decompressed data is non-null
    if (this.decompressedData === null)
      throw new Error("png 'decompressedData' shouldn't be null after calling decompress method to deflate data");
    const scanlineLength = this.nbytesPerPixel * this.width;
    let row = 0;
    let c = offset;
    while (row < this.height && this.dpos < this.decompressedData.length) {
        for (let i = 0; i < scanlineLength; i++)
          outputArr[c++] = this.decompressedData[this.dpos++] / 255;
        row++;
    }
  }
}