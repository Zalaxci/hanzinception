import { PNGModel } from "./PNG";

const HANZI_LIST = ["一", "九", "人", "做", "几", "叫", "商", "多", "字", "年", "时", "本", "气", "狗", "米", "见", "谁", "院", "七", "习", "什", "儿", "出", "吃", "喂", "大", "学", "店", "明", "机", "水", "猫", "系", "视", "谢", "雨", "三", "书", "今", "先", "分", "同", "喜", "天", "客", "开", "星", "来", "汉", "现", "老", "觉", "起", "零", "上", "买", "他", "八", "前", "名", "喝", "太", "家", "影", "昨", "杯", "没", "生", "能", "认", "车", "面", "下", "了", "们", "六", "北", "后", "四", "女", "对", "很", "是", "果", "漂", "电", "脑", "识", "这", "飞", "不", "二", "会", "关", "医", "吗", "回", "她", "小", "怎", "月", "校", "火", "的", "苹", "话", "那", "饭", "东", "五", "住", "兴", "十", "听", "国", "好", "少", "想", "有", "样", "点", "看", "茶", "语", "都", "馆", "个", "些", "作", "再", "午", "呢", "在", "妈", "岁", "我", "朋", "桌", "热", "睡", "菜", "说", "里", "高", "中", "京", "你", "写", "去", "和", "坐", "姐", "工", "打", "服", "椅", "爱", "租", "衣", "请", "钟", "么", "亮", "候", "冷", "友", "哪", "块", "子", "师", "日", "期", "欢", "爸", "站", "西", "读", "钱"];
export const N_TRAIN_IMAGES = 580;
export const N_TEST_IMAGES = 135;
const DOWNLOAD_BASE_PATH = "assets/chinese-handwriting";

export function getTotalHanziCount(): number {
    return HANZI_LIST.length;
}
export function getTotalImageCount(): number {
    return HANZI_LIST.length * (N_TRAIN_IMAGES + N_TEST_IMAGES);
}
export function getHanzi(i: number): string {
    return HANZI_LIST[i];
}
export function getImageSize(): number {
    return 50;
}

export interface HanziStorageModel {
    readFile(hanzi: string, imageNumber: number, isTestData: boolean): Promise<PNGModel>
    downloadAll(handleSuccess: (n: number) => void, handleError: (err: any) => void): void
}
export async function createHanziStorageModel(): Promise<HanziStorageModel> {
    const rootDir = await navigator.storage.getDirectory();
    const hanziDir = await rootDir.getDirectoryHandle("hanzi");
    const trainDir = await hanziDir.getDirectoryHandle("Train");
    const testDir = await hanziDir.getDirectoryHandle("Test");
    // Test if file with filename exists in specific directory
    async function exists(directory: FileSystemDirectoryHandle, filename: string, handleError: (err: any) => void): Promise<boolean> {
        try {
            await directory.getFileHandle(filename, { create: false });
            return true;
        } catch(err) {
            if (err instanceof DOMException && err.name === "NotFoundError")
                return false;
            handleError(err);
            return false;
        }
    }
    // Download individual hanzi file
    async function downloadFile(currHanziDir: FileSystemDirectoryHandle, hanzi: string, imageNumber: number, isTestData: boolean = false) {
        const resp = await fetch(`${DOWNLOAD_BASE_PATH}/${isTestData? "Test" : "Train"}/${hanzi}/${imageNumber}.png`);
        const imgData = await resp.bytes();
        const imgHandle = await currHanziDir.getFileHandle(`${imageNumber}.png`);
        const imgWriter = await imgHandle.createWritable();
        await imgWriter.write(imgData);
    }
    return {
        async readFile(hanzi: string, imageNumber: number, isTestData: boolean = false): Promise<PNGModel> {
            const currHanziDir = isTestData? await testDir.getDirectoryHandle(hanzi) : await trainDir.getDirectoryHandle(hanzi);
            const imgHandle = await currHanziDir.getFileHandle(`${imageNumber}.png`);
            const imgFile = await imgHandle.getFile();
            const imgBytes = new Uint8Array(await imgFile.arrayBuffer());
            const png = new PNGModel(imgBytes);
            return png;
        },
        async downloadAll(handleSuccess: (n: number) => void, handleError: (err: any) => void) {
            if (await exists(hanziDir, "ALL_DOWNLOADED", handleError)) {
                handleSuccess(getTotalImageCount());
                return;
            };
            let promises = [];
            // Download & store training data
            for (let hanzi of HANZI_LIST) {
                const currHanziDir = await trainDir.getDirectoryHandle(hanzi);
                for (let imageNumber = 1; imageNumber <= N_TRAIN_IMAGES; imageNumber++) {
                    if (await exists(currHanziDir, `${imageNumber}.png`, handleError)) handleSuccess(1);
                    else promises.push(
                        downloadFile(currHanziDir, hanzi, imageNumber).then(() => handleSuccess(1)).catch(handleError)
                    );
                    if (promises.length === 200) {
                        await Promise.all(promises);
                        promises = [];
                    }
                }
            }
            // Download & store test data
            for (let hanzi of HANZI_LIST) {
                const currHanziDir = await testDir.getDirectoryHandle(hanzi);
                for (let imageNumber = 1; imageNumber <= N_TEST_IMAGES; imageNumber++) {
                    if (await exists(currHanziDir, `${imageNumber}.png`, handleError)) handleSuccess(1);
                    else promises.push(
                        downloadFile(currHanziDir, hanzi, imageNumber, true).then(() => handleSuccess(1)).catch(handleError)
                    );
                    if (promises.length === 200) {
                        await Promise.all(promises);
                        promises = [];
                    }
                }
            }
            await Promise.all(promises);
        },
    }
}