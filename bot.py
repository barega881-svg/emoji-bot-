# language: Python, file: bot.py, target: Render Docker Web Service, Python 3.11
import asyncio
import os
import subprocess
import tempfile
import shutil

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, FSInputFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============ НАСТРОЙКИ ============
BOT_TOKEN = os.environ["BOT_TOKEN"]

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

class Flow(StatesGroup):
    waiting_emoji = State()
    waiting_text = State()

def menu():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="🎨 Сделать эмодзи")]],
        resize_keyboard=True
    )

# ============ ОБРАБОТКА ВИДЕО ============

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        FFMPEG, '-i', video_path,
        '-vf', 'fps=30,scale=100:100:force_original_aspect_ratio=decrease,pad=100:100:(ow-iw)/2:(oh-ih)/2',
        os.path.join(output_dir, 'frame_%04d.png'),
        '-y'
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def find_text_bboxes_simple(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = gray.shape
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if 5 < cw < w * 0.9 and 5 < ch < h * 0.5 and cw * ch > 40:
            boxes.append({'x1': x, 'y1': y, 'x2': x + cw, 'y2': y + ch})
    return boxes

def find_text_bboxes(frame):
    try:
        import pytesseract
        data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT, lang='rus+eng')
        boxes = []
        for i, text in enumerate(data['text']):
            if text.strip() and int(data['conf'][i]) > 30:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                boxes.append({'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})
        return boxes
    except Exception:
        return find_text_bboxes_simple(frame)

def inpaint_text(frame, boxes):
    if not boxes:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for box in boxes:
        cv2.rectangle(mask, (box['x1']-2, box['y1']-2), (box['x2']+2, box['y2']+2), 255, -1)
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

def draw_new_text(frame, text, boxes):
    if not boxes:
        h, w = frame.shape[:2]
        boxes = [{'x1': int(w*0.1), 'y1': int(h*0.35), 'x2': int(w*0.9), 'y2': int(h*0.65)}]
    
    x1 = min(b['x1'] for b in boxes)
    y1 = min(b['y1'] for b in boxes)
    x2 = max(b['x2'] for b in boxes)
    y2 = max(b['y2'] for b in boxes)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_size = max(8, (y2 - y1) // 2)
    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((0, 0), text, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 - th) // 2

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            draw.text((tx+dx, ty+dy), text, font=font, fill=(0, 0, 0))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def frames_to_webm(frames_dir, output_path, fps=30):
    cmd = [
        FFMPEG,
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libvpx-vp9',
        '-pix_fmt', 'yuva420p',
        '-b:v', '0',
        '-crf', '45',
        '-an',
        '-t', '3',
        output_path,
        '-y'
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def process_emoji(input_webm, new_text, output_webm):
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = os.path.join(tmpdir, 'frames')
        out_frames_dir = os.path.join(tmpdir, 'out_frames')
        os.makedirs(out_frames_dir, exist_ok=True)

        extract_frames(input_webm, frames_dir)
        frame_files = sorted(os.listdir(frames_dir))
        if not frame_files:
            raise Exception("FFmpeg не извлёк кадры. Проверь, что .webm читается.")

        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        if first_frame is None:
            raise Exception("Не удалось прочитать первый кадр")

        reference_boxes = find_text_bboxes(first_frame)

        for fname in frame_files:
            frame = cv2.imread(os.path.join(frames_dir, fname))
            if frame is None:
                continue
            boxes = reference_boxes if reference_boxes else find_text_bboxes(frame)
            cleaned = inpaint_text(frame, boxes)
            result = draw_new_text(cleaned, new_text, boxes)
            cv2.imwrite(os.path.join(out_frames_dir, fname), result)

        frames_to_webm(out_frames_dir, output_webm)
    return output_webm

# ============ ХЕНДЛЕРЫ БОТА ============

@dp.message(CommandStart())
async def start(m: Message, state: FSMContext):
    await state.clear()
    await m.answer(
        "Привет! Отправь мне видео-эмодзи (.webm) или видео-стикер, "
        "и я заменю в нём текст на твой.\n\n"
        "⚠️ Готовый эмодзи можно использовать только с Telegram Premium.",
        reply_markup=menu()
    )

@dp.message(F.text == "🎨 Сделать эмодзи")
async def ask_emoji(m: Message, state: FSMContext):
    await state.set_state(Flow.waiting_emoji)
    await m.answer("Отправь .webm-файл или анимированный видео-стикер.")

@dp.message(Flow.waiting_emoji, F.document)
async def got_doc(m: Message, state: FSMContext):
    if not m.document.file_name.lower().endswith('.webm'):
        await m.answer("Нужен .webm файл.")
        return
    await state.update_data(file_id=m.document.file_id)
    await state.set_state(Flow.waiting_text)
    await m.answer("Теперь напиши текст, который вставить в эмодзи:")

@dp.message(Flow.waiting_emoji, F.sticker)
async def got_sticker(m: Message, state: FSMContext):
    if not m.sticker.is_video:
        await m.answer("Нужен именно анимированный (видео) стикер/эмодзи.")
        return
    await state.update_data(file_id=m.sticker.file_id)
    await state.set_state(Flow.waiting_text)
    await m.answer("Теперь напиши текст, который вставить в эмодзи:")

@dp.message(Flow.waiting_text)
async def got_text(m: Message, state: FSMContext):
    new_text = m.text.strip()
    if not new_text:
        await m.answer("Текст пустой.")
        return

    data = await state.get_data()
    file_id = data['file_id']
    status = await m.answer("⏳ Скачиваю и обрабатываю эмодзи... Это может занять пару минут.")

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, 'input.webm')
        output_path = os.path.join(tmp, 'output.webm')

        try:
            file = await bot.get_file(file_id)
            await bot.download_file(file.file_path, input_path)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, process_emoji, input_path, new_text, output_path)

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Файл не создался (возможно, FFmpeg выдал ошибку).")

            video = FSInputFile(output_path, filename='emoji.webm')
            await m.answer_video(
                video,
                caption=f"✅ Готово! Текст: «{new_text}»\n\n"
                        f"⚠️ Для использования как эмодзи нужен Telegram Premium.",
            )
        except Exception as e:
            await m.answer(f"❌ Ошибка:\n<code>{str(e)[:400]}</code>")
        finally:
            await status.delete()

    await state.clear()
    await m.answer("Можешь отправить ещё один эмодзи.", reply_markup=menu())

# ============ ВЕБ-СЕРВЕР ДЛЯ RENDER KEEP-ALIVE ============

async def health_server():
    from aiohttp import web
    async def health(request):
        return web.Response(text="OK")
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/", health)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Health server on port {port}")

async def main():
    print("Бот запущен...")
    await asyncio.gather(
        health_server(),
        dp.start_polling(bot),
    )

if __name__ == "__main__":
    asyncio.run(main())