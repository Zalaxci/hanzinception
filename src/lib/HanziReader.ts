import { numpy as np, tree } from "@jax-js/jax";
import PNGParser from "./PNG";

export const hanziList = ["一", "九", "人", "做", "几", "叫", "商", "多", "字", "年", "时", "本", "气", "狗", "米", "见", "谁", "院", "七", "习", "什", "儿", "出", "吃", "喂", "大", "学", "店", "明", "机", "水", "猫", "系", "视", "谢", "雨", "三", "书", "今", "先", "分", "同", "喜", "天", "客", "开", "星", "来", "汉", "现", "老", "觉", "起", "零", "上", "买", "他", "八", "前", "名", "喝", "太", "家", "影", "昨", "杯", "没", "生", "能", "认", "车", "面", "下", "了", "们", "六", "北", "后", "四", "女", "对", "很", "是", "果", "漂", "电", "脑", "识", "这", "飞", "不", "二", "会", "关", "医", "吗", "回", "她", "小", "怎", "月", "校", "火", "的", "苹", "话", "那", "饭", "东", "五", "住", "兴", "十", "听", "国", "好", "少", "想", "有", "样", "点", "看", "茶", "语", "都", "馆", "个", "些", "作", "再", "午", "呢", "在", "妈", "岁", "我", "朋", "桌", "热", "睡", "菜", "说", "里", "高", "中", "京", "你", "写", "去", "和", "坐", "姐", "工", "打", "服", "椅", "爱", "租", "衣", "请", "钟", "么", "亮", "候", "冷", "友", "哪", "块", "子", "师", "日", "期", "欢", "爸", "站", "西", "读", "钱"];
export const imageCountPerHanzi = 580; // TODO: Make this different for each hanzi, according to folder structure

// Create array with number from 1 to N in random order utilizing only one array, the one returned, and no memory reallocation
export function randomlyOrderNumbers1toN(arr: Uint16Array) {
    // Set entire array to 0
    for (let i = 0; i < arr.length; i++)
        arr[i] = 0;
    // Repeat for all numbers from 1 to N
    for (let i = 1; i <= arr.length; i++) {
        // Peak random position to place
        let posToAdd = Math.floor(Math.random() * arr.length);
        // If already occupied, move to next position, cycling back to 0 once last index is reached, to find where to place
        while (arr[posToAdd] !== 0) {
            if (posToAdd === arr.length - 1) posToAdd = 0;
            else posToAdd++;
        }
        // Place "i", i.e. the current index, in specified position
        arr[posToAdd] = i;
    }
}

export async function* hanziImageIterator(w: number, h: number, nepochs: number, invert: boolean = false): AsyncGenerator<np.Array> {
    const npixels = w * h;
    const buffer = new Float32Array(hanziList.length * w * h);
    const imageIDs = new Uint16Array(imageCountPerHanzi);
    // OPFS
    const rootDir = await navigator.storage.getDirectory();
    const baseDir = await rootDir.getDirectoryHandle("hanzi");
    const trainDir = await baseDir.getDirectoryHandle("Train");
    const imgFiles: FileSystemFileHandle[] = new Array(hanziList.length);
    // Jax.js Arrays
    const batches: (np.Array | undefined)[] = new Array(imageCountPerHanzi);
    for (let iepoch = 0; iepoch < nepochs; iepoch++) {
        randomlyOrderNumbers1toN(imageIDs);
        for (let imgID of imageIDs) {
            if (batches[imgID]) {
                yield batches[imgID].ref;
                continue;
            }
            for (let ihanzi = 0; ihanzi < hanziList.length; ihanzi++) {
                const hanziDir = await trainDir.getDirectoryHandle(hanziList[ihanzi]);
                imgFiles[ihanzi] = await hanziDir.getFileHandle(`${imgID}.png`);
            }
            await Promise.all(
                imgFiles.map(async (fileHandle, i) => {
                    const file = await fileHandle.getFile();
                    const bytes = new Uint8Array(await file.arrayBuffer());
                    const png = new PNGParser(w, h, bytes);
                    await png.decompressAndDecodePixels(buffer, i * npixels);
                })
            );
            batches[imgID] = np.array(buffer, { shape: [hanziList.length, npixels] });
            yield batches[imgID].ref;
        }
    }
    tree.dispose(batches);
}

// TODO: For greater indices we should have only the hanzi that contain images with said index in the labels
export async function* hanziLabelIterator(nepochs: number): AsyncGenerator<np.Array> {
    const labels = np.arange(hanziList.length);
    const iters = nepochs * imageCountPerHanzi;
    for (let i = 0; i < iters; i++)
        yield labels.ref;
    labels.dispose();
}