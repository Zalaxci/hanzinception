import { assert, expect, test } from 'vitest'
import PNGParser from './PNG'

test("parses IDAT succesfully", () => {
    const fakePngData = new TextEncoder().encode("________\0\0\0\0IDATbullshitpng UNRELATED DATA IN THE END");
    fakePngData[11] = "bullshitpng".length;
    const png = new PNGParser(300, 400, fakePngData);
    const extractedIDAT = png.imgData.slice(png.filePos, png.filePos + png.idatLength);
    const str = new TextDecoder().decode(extractedIDAT);
    expect(str).toBe("bullshitpng");
})
test("fetches PNG & retreives image data from PNG file correctly", async () => {
    const png = await PNGParser.fromURL(50, 50, "http://localhost:5173/assets/chinese-handwriting/Test/一/42.png");
    expect(png.imgData[png.filePos - 1], "before first byte you have IDAT").toBe(0x54); // it should be T
    expect(png.imgData[png.filePos], "first byte is correct").toBe(0x48);
    expect(png.imgData[png.filePos + 1], "second byte is correct").toBe(0xC7);
    expect(png.imgData[png.filePos + 2], "third byte is correct").toBe(0xED);
    expect(png.idatLength, "IDAT section length is correct").toBe(485 - 12); // IDAT section size is 485 according to https://omniviewer.org/ but 4 is IEND, 4 is CRC check and 4 is stuff after IEND
})
test("decompressing PNG data resolves without error", async () => {
    const png = await PNGParser.fromURL(50, 50, "http://localhost:5173/assets/chinese-handwriting/Train/%E8%AF%BB/20.png");
    await png.decompress();
})
test("pixels of 20th and 30th row are correctly decoded", async () => {
    const row20Reference = new Uint8Array([255, 255, 255, 255, 250, 218, 209, 212, 216, 190, 136, 129, 111,
                                            90,  83,  90,  95,  87,  84,  91,  85,  52,  29,  41,  40,  31,
                                            25,  23,  25,  36,  36,  31,  39,  71,  78,  73,  83,  95,  82,
                                            81, 109, 171, 214, 222, 255, 255, 255, 255, 255, 255]);
    const row30Reference = new Uint8Array([ 96,  67,  53,  41,  23,  18,  15,  14,  16,  22,  36,  52,  47,
                                            39,  50,  74,  81,  76,  83,  87,  94, 105,  92,  85, 105, 131,
                                           145, 153, 178, 177, 160, 152, 150, 123,  92,  72,  78,  95,  93,
                                            72,  69,  76,  74,  68,  84,  91,  70,  59,  79, 109 ]);
    const png = await PNGParser.fromURL(50, 50, "http://localhost:5173/assets/chinese-handwriting/Test/一/42.png");
    const pixels = new Float32Array(2500);
    await png.decompressAndDecodePixels(pixels);
    for (let i = 1; i < 50; i++)
        expect(pixels[20*50 + i], "20th row pixels should match what you get from python with opencv").toBeCloseTo(row20Reference[i]/255);
    for (let i = 1; i < 50; i++)
        expect(pixels[25*50 + i], "30th row pixels should match what you get from python with opencv").toBeCloseTo(row30Reference[i]/255);
})