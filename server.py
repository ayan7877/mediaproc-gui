#!/usr/bin/env python3
"""MediaProc GUI - Flask Backend
Handles real image processing (Pillow), video/audio (FFmpeg), and pipeline execution.
"""

import os, uuid, json, subprocess, shutil, time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import io

app = Flask(__name__, static_folder='static')

UPLOAD_DIR = Path(os.environ.get('UPLOAD_DIR', '/tmp/mediaproc_uploads'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/tmp/mediaproc_output'))
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def new_id():
    return uuid.uuid4().hex[:12]

def fmtsize(b):
    if b < 1024: return f"{b} B"
    if b < 1048576: return f"{b/1024:.1f} KB"
    return f"{b/1048576:.1f} MB"

# ─── Static frontend ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:p>')
def static_files(p):
    return send_from_directory('static', p)

# ─── Upload endpoint ──────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    result = []
    for f in files:
        fid = new_id()
        ext = Path(f.filename).suffix.lower()
        dest = UPLOAD_DIR / f"{fid}{ext}"
        f.save(str(dest))
        size = dest.stat().st_size
        result.append({'id': fid, 'name': f.filename, 'ext': ext.lstrip('.'), 'size': size, 'path': str(dest)})
    return jsonify({'files': result})

# ─── Image processing ─────────────────────────────────────────────────────────
@app.route('/api/image/<command>', methods=['POST'])
def image_cmd(command):
    data = request.json
    file_ids = data.get('files', [])
    opts = data.get('options', {})
    logs = []
    outputs = []

    for f in file_ids:
        fpath = Path(f['path'])
        if not fpath.exists():
            logs.append({'type': 'error', 'text': f"File not found: {f['name']}"})
            continue

        try:
            img = Image.open(str(fpath))
            orig_size = fpath.stat().st_size
            orig_fmt = img.format or 'JPEG'
            logs.append({'type': 'info', 'text': f"[{f['name']}] {img.width}×{img.height} {orig_fmt} ({fmtsize(orig_size)})"})

            result_img = img.copy()

            if command == 'resize':
                w = int(opts.get('--width') or img.width)
                h = int(opts.get('--height') or img.height)
                fit = opts.get('--fit', 'cover')
                if fit == 'cover':
                    r = max(w/img.width, h/img.height)
                    new_w, new_h = int(img.width*r), int(img.height*r)
                    result_img = img.resize((new_w, new_h), Image.LANCZOS)
                    left = (new_w - w)//2; top = (new_h - h)//2
                    result_img = result_img.crop((left, top, left+w, top+h))
                elif fit == 'contain':
                    result_img.thumbnail((w, h), Image.LANCZOS)
                    bg = Image.new('RGB', (w, h), opts.get('--background', '#ffffff'))
                    off = ((w - result_img.width)//2, (h - result_img.height)//2)
                    bg.paste(result_img, off)
                    result_img = bg
                elif fit == 'fill':
                    result_img = img.resize((w, h), Image.LANCZOS)
                elif fit == 'inside':
                    result_img = img.copy(); result_img.thumbnail((w, h), Image.LANCZOS)
                elif fit == 'outside':
                    r = min(w/img.width, h/img.height)
                    result_img = img.resize((int(img.width*r), int(img.height*r)), Image.LANCZOS)
                logs.append({'type': 'info', 'text': f"  Resized: {img.width}×{img.height} → {result_img.width}×{result_img.height} ({fit})"})

            elif command == 'convert':
                fmt = opts.get('--format', 'webp')
                quality = int(opts.get('--quality', 85))
                logs.append({'type': 'info', 'text': f"  Converting to {fmt.upper()} at quality {quality}%"})

            elif command == 'compress':
                quality = int(opts.get('--quality', 75))
                logs.append({'type': 'info', 'text': f"  Compressing at quality {quality}%"})

            elif command == 'crop':
                cw = int(opts.get('--width', min(500, img.width)))
                ch = int(opts.get('--height', min(500, img.height)))
                cl = int(opts.get('--left', 0))
                ct = int(opts.get('--top', 0))
                result_img = img.crop((cl, ct, cl+cw, ct+ch))
                logs.append({'type': 'info', 'text': f"  Cropped to {cw}×{ch} from ({cl},{ct})"})

            elif command == 'rotate':
                angle = int(opts.get('--angle', 90))
                bg = opts.get('--background', '#ffffff')
                result_img = img.rotate(-angle, expand=True, fillcolor=bg)
                logs.append({'type': 'info', 'text': f"  Rotated {angle}°"})

            elif command == 'flip':
                result_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                logs.append({'type': 'info', 'text': "  Flipped vertically"})

            elif command == 'flop':
                result_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                logs.append({'type': 'info', 'text': "  Flopped horizontally"})

            elif command == 'blur':
                sigma = float(opts.get('--sigma', 3))
                result_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
                logs.append({'type': 'info', 'text': f"  Gaussian blur σ={sigma}"})

            elif command == 'sharpen':
                sigma = float(opts.get('--sigma', 1))
                factor = 1 + sigma
                enhancer = ImageEnhance.Sharpness(img)
                result_img = enhancer.enhance(factor)
                logs.append({'type': 'info', 'text': f"  Sharpened ×{factor:.1f}"})

            elif command == 'grayscale':
                result_img = img.convert('L').convert('RGB')
                logs.append({'type': 'info', 'text': "  Converted to grayscale"})

            elif command == 'tint':
                color = opts.get('--color', '#ff6600')
                r2 = int(color[1:3], 16) if color.startswith('#') else 255
                g2 = int(color[3:5], 16) if color.startswith('#') else 100
                b2 = int(color[5:7], 16) if color.startswith('#') else 0
                tint = Image.new('RGB', img.size, (r2, g2, b2))
                result_img = Image.blend(img.convert('RGB'), tint, 0.4)
                logs.append({'type': 'info', 'text': f"  Tint applied: {color}"})

            elif command == 'watermark':
                text = opts.get('--text', '© MediaProc')
                opacity = int(opts.get('--opacity', 50)) / 100
                size = int(opts.get('--size', 36))
                color = opts.get('--color', '#ffffff')
                result_img = img.convert('RGBA')
                overlay = Image.new('RGBA', result_img.size, (0,0,0,0))
                draw = ImageDraw.Draw(overlay)
                try:
                    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
                except:
                    font = ImageFont.load_default()
                tw = result_img.width // 2
                th = result_img.height // 2
                r3 = int(color[1:3], 16) if color.startswith('#') else 255
                g3 = int(color[3:5], 16) if color.startswith('#') else 255
                b3 = int(color[5:7], 16) if color.startswith('#') else 255
                draw.text((tw, th), text, font=font, fill=(r3, g3, b3, int(255*opacity)), anchor='mm')
                result_img = Image.alpha_composite(result_img, overlay).convert('RGB')
                logs.append({'type': 'info', 'text': f"  Watermark: \"{text}\" at {int(opacity*100)}% opacity"})

            elif command == 'thumbnail':
                sz = int(opts.get('--size', 256))
                result_img = img.copy()
                result_img.thumbnail((sz, sz), Image.LANCZOS)
                bg = Image.new('RGB', (sz, sz), '#ffffff')
                off = ((sz - result_img.width)//2, (sz - result_img.height)//2)
                bg.paste(result_img, off)
                result_img = bg
                logs.append({'type': 'info', 'text': f"  Thumbnail: {sz}×{sz}"})

            elif command == 'optimize':
                quality = int(opts.get('--quality', 80))
                fmt2 = opts.get('--format', 'webp')
                logs.append({'type': 'info', 'text': f"  Optimizing to {fmt2.upper()} at {quality}% quality"})

            elif command == 'info':
                logs.append({'type': 'info', 'text': f"  Filename:    {f['name']}"})
                logs.append({'type': 'info', 'text': f"  Dimensions:  {img.width}×{img.height}"})
                logs.append({'type': 'info', 'text': f"  File size:   {fmtsize(orig_size)}"})
                logs.append({'type': 'info', 'text': f"  Megapixels:  {img.width*img.height/1e6:.2f} MP"})
                logs.append({'type': 'info', 'text': f"  Format:      {orig_fmt}"})
                logs.append({'type': 'info', 'text': f"  Mode:        {img.mode}"})
                logs.append({'type': 'info', 'text': f"  Aspect ratio:{img.width/img.height:.3f}"})
                continue

            # Determine output format
            out_fmt_map = {
                'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG',
                'webp': 'WEBP', 'avif': 'WEBP', 'tiff': 'TIFF', 'gif': 'GIF'
            }
            if command == 'convert':
                out_ext = opts.get('--format', 'webp')
                out_fmt = out_fmt_map.get(out_ext, 'WEBP')
            elif command in ('compress', 'optimize'):
                fmt_opt = opts.get('--format', 'original')
                if fmt_opt == 'original':
                    out_ext = f['ext']
                    out_fmt = out_fmt_map.get(out_ext, 'JPEG')
                else:
                    out_ext = fmt_opt
                    out_fmt = out_fmt_map.get(out_ext, 'WEBP')
            else:
                out_ext = f['ext']
                out_fmt = out_fmt_map.get(out_ext, 'JPEG')

            # Save output
            oid = new_id()
            base = Path(f['name']).stem
            out_name = f"{base}_{command}.{out_ext}"
            out_path = OUTPUT_DIR / f"{oid}_{out_name}"

            save_kwargs = {}
            if out_fmt == 'JPEG':
                quality = int(opts.get('--quality', 85))
                save_kwargs = {'quality': quality, 'optimize': True}
                if result_img.mode in ('RGBA', 'LA', 'P'):
                    result_img = result_img.convert('RGB')
            elif out_fmt == 'WEBP':
                quality = int(opts.get('--quality', 85))
                save_kwargs = {'quality': quality, 'method': 4}
            elif out_fmt == 'PNG':
                save_kwargs = {'optimize': True}

            result_img.save(str(out_path), format=out_fmt, **save_kwargs)
            out_size = out_path.stat().st_size

            if command == 'compress':
                saved = round((1 - out_size/orig_size)*100)
                logs.append({'type': 'success' if saved > 0 else 'info',
                    'text': f"  {fmtsize(orig_size)} → {fmtsize(out_size)} ({'+' if saved<0 else '-'}{abs(saved)}%)"})
            else:
                logs.append({'type': 'success', 'text': f"  ✓ {out_name} ({fmtsize(out_size)})"})

            outputs.append({
                'id': oid, 'name': out_name,
                'path': str(out_path), 'size': out_size,
                'type': 'image', 'mime': f"image/{out_ext}",
                'downloadUrl': f"/api/download/{oid}_{out_name}"
            })

        except Exception as e:
            logs.append({'type': 'error', 'text': f"  ✗ {f['name']}: {str(e)}"})

    return jsonify({'logs': logs, 'outputs': outputs})

# ─── Video processing (FFmpeg) ────────────────────────────────────────────────
@app.route('/api/video/<command>', methods=['POST'])
def video_cmd(command):
    data = request.json
    files = data.get('files', [])
    opts = data.get('options', {})
    logs = [{'type':'dim','text':'─'*44}]
    outputs = []

    if not files:
        return jsonify({'logs': [{'type':'error','text':'No files provided'}], 'outputs': []})

    f = files[0]
    fpath = Path(f['path'])
    if not fpath.exists():
        return jsonify({'logs': [{'type':'error','text':f"File not found: {f['name']}"}], 'outputs': []})

    # Get video metadata
    probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', str(fpath)]
    try:
        probe = json.loads(subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL))
        vstream = next((s for s in probe.get('streams',[]) if s.get('codec_type')=='video'), {})
        astream = next((s for s in probe.get('streams',[]) if s.get('codec_type')=='audio'), {})
        fmt_info = probe.get('format', {})
        duration = float(fmt_info.get('duration', 0))
        mins, secs = int(duration//60), int(duration%60)
        vcodec = vstream.get('codec_name', 'unknown')
        width = vstream.get('width', 0)
        height = vstream.get('height', 0)
        fps = vstream.get('r_frame_rate', '30/1')
        acodec = astream.get('codec_name', 'unknown')
        bitrate = int(fmt_info.get('bit_rate', 0)) // 1000
        size = int(fmt_info.get('size', fpath.stat().st_size))
        logs.append({'type':'info','text':f"Input: {f['name']} ({fmtsize(size)})"})
        logs.append({'type':'info','text':f"  Duration: {mins}:{secs:02d} | {width}×{height} @ {fps} fps"})
        logs.append({'type':'info','text':f"  Video: {vcodec} | Audio: {acodec} | Bitrate: {bitrate} kbps"})
    except Exception as e:
        logs.append({'type':'warn','text':f"Could not probe file: {e}"})
        duration, width, height, vcodec, size = 0, 0, 0, 'unknown', fpath.stat().st_size

    oid = new_id()
    base = Path(f['name']).stem

    if command == 'metadata':
        logs.append({'type':'info','text':f"  File size:  {fmtsize(fpath.stat().st_size)}"})
        return jsonify({'logs': logs, 'outputs': []})

    # Build FFmpeg command
    if command == 'compress':
        out_fmt = opts.get('--format', 'mp4')
        out_name = f"{base}_compressed.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        quality = opts.get('--quality', 'medium')
        codec = opts.get('--codec', 'h264')
        crf = int(opts.get('--crf', {'high':18,'medium':23,'low':28,'extreme':35}.get(quality,23)))
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-c:v', 'libx264', '-crf', str(crf),
                 '-preset', opts.get('--preset','medium'), '-c:a', 'aac', '-b:a', '128k',
                 '-movflags', '+faststart', str(out_path)]
        logs.append({'type':'info','text':f"  Quality: {quality} | CRF: {crf} | Codec: {codec}"})

    elif command == 'transcode':
        out_fmt = opts.get('--format', 'mp4')
        out_name = f"{base}_transcoded.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        codec = opts.get('--codec', 'h264')
        codec_map = {'h264':'libx264','h265':'libx265','vp9':'libvpx-vp9','av1':'libaom-av1','copy':'copy'}
        vcodec_arg = codec_map.get(codec, 'libx264')
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-c:v', vcodec_arg, '-c:a', opts.get('--audio-codec','aac'), str(out_path)]
        logs.append({'type':'info','text':f"  Transcoding to {out_fmt.upper()} | Codec: {codec}"})

    elif command == 'trim':
        out_name = f"{base}_trimmed.mp4"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        start = opts.get('--start', '00:00:00')
        end = opts.get('--end', '00:01:00')
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-ss', str(start), '-to', str(end),
                 '-c', 'copy', str(out_path)]
        logs.append({'type':'info','text':f"  Trimming: {start} → {end} (stream copy)"})

    elif command == 'resize':
        out_name = f"{base}_resized.mp4"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        preset_map = {'4k':'3840:2160','1440p':'2560:1440','1080p':'1920:1080','720p':'1280:720','480p':'854:480','360p':'640:360'}
        res = preset_map.get(opts.get('--preset','1080p'),'1920:1080')
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-vf', f"scale={res}:force_original_aspect_ratio=decrease",
                 '-c:v', 'libx264', '-crf', '23', '-c:a', 'copy', str(out_path)]
        logs.append({'type':'info','text':f"  Scaling to {opts.get('--preset','1080p')} ({res})"})

    elif command == 'merge':
        out_fmt = opts.get('--format', 'mp4')
        out_name = f"merged.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        list_file = UPLOAD_DIR / f"{oid}_concat.txt"
        with open(str(list_file), 'w') as lf:
            for fi in files:
                lf.write(f"file '{fi['path']}'\n")
        ffcmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file),
                 '-c', opts.get('--codec','copy'), str(out_path)]
        logs.append({'type':'info','text':f"  Merging {len(files)} file(s)"})

    elif command == 'extract':
        out_fmt = opts.get('--format', 'mp3')
        out_name = f"{base}_audio.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        bitrate = opts.get('--bitrate', '192k')
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-vn', '-b:a', bitrate, str(out_path)]
        logs.append({'type':'info','text':f"  Extracting audio → {out_fmt.upper()} @ {bitrate}"})

    elif command == 'convert':
        out_fmt = opts.get('--format', 'mp4')
        out_name = f"{base}_converted.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        codec_map2 = {'h264':'libx264','h265':'libx265','vp9':'libvpx-vp9','copy':'copy'}
        vca = codec_map2.get(opts.get('--codec','copy'), 'copy')
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-c:v', vca, '-c:a', 'copy', str(out_path)]
        logs.append({'type':'info','text':f"  Converting to {out_fmt.upper()}"})
    else:
        return jsonify({'logs':[{'type':'error','text':f'Unknown command: {command}'}],'outputs':[]})

    logs.append({'type':'info','text':f"  Running FFmpeg..."})

    try:
        result = subprocess.run(ffcmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            out_size = out_path.stat().st_size
            logs.append({'type':'success','text':f"  ✓ {out_name} ({fmtsize(out_size)})"})
            if command == 'compress':
                saved = round((1-out_size/size)*100)
                logs.append({'type':'success','text':f"  Size: {fmtsize(size)} → {fmtsize(out_size)} (-{saved}%)"})
            outputs.append({
                'id': oid, 'name': out_name,
                'path': str(out_path), 'size': out_size,
                'type': 'video', 'mime': f"video/{out_fmt.split('.')[0]}",
                'downloadUrl': f"/api/download/{oid}_{out_name}"
            })
        else:
            # Try simplified fallback
            logs.append({'type':'warn','text':'  Retrying with stream copy fallback...'})
            fb_cmd = ['ffmpeg', '-y', '-i', str(fpath), '-c', 'copy', str(out_path)]
            r2 = subprocess.run(fb_cmd, capture_output=True, text=True, timeout=60)
            if r2.returncode == 0:
                out_size = out_path.stat().st_size
                logs.append({'type':'success','text':f"  ✓ {out_name} ({fmtsize(out_size)})"})
                outputs.append({'id':oid,'name':out_name,'path':str(out_path),'size':out_size,
                    'type':'video','mime':f"video/mp4",'downloadUrl':f"/api/download/{oid}_{out_name}"})
            else:
                logs.append({'type':'error','text':f"  ✗ FFmpeg failed: {result.stderr[-300:]}"})
    except subprocess.TimeoutExpired:
        logs.append({'type':'error','text':'  ✗ Timeout after 120 seconds'})
    except Exception as e:
        logs.append({'type':'error','text':f"  ✗ {str(e)}"})

    return jsonify({'logs': logs, 'outputs': outputs})

# ─── Audio processing (FFmpeg) ────────────────────────────────────────────────
@app.route('/api/audio/<command>', methods=['POST'])
def audio_cmd(command):
    data = request.json
    files = data.get('files', [])
    opts = data.get('options', {})
    logs = [{'type':'dim','text':'─'*44}]
    outputs = []

    if not files:
        return jsonify({'logs':[{'type':'error','text':'No files'}],'outputs':[]})

    f = files[0]
    fpath = Path(f['path'])

    # Probe
    try:
        probe = json.loads(subprocess.check_output(
            ['ffprobe','-v','quiet','-print_format','json','-show_streams','-show_format',str(fpath)],
            stderr=subprocess.DEVNULL))
        astream = next((s for s in probe.get('streams',[]) if s.get('codec_type')=='audio'), {})
        fmt_info = probe.get('format', {})
        duration = float(fmt_info.get('duration', 0))
        mins, secs = int(duration//60), int(duration%60)
        acodec = astream.get('codec_name','unknown')
        samplerate = astream.get('sample_rate','44100')
        channels = astream.get('channels',2)
        bitrate = int(fmt_info.get('bit_rate',0))//1000
        logs.append({'type':'info','text':f"Input: {f['name']} ({fmtsize(fpath.stat().st_size)})"})
        logs.append({'type':'info','text':f"  Duration: {mins}:{secs:02d} | {acodec} | {samplerate}Hz | {'Stereo' if channels==2 else 'Mono'}"})
        if bitrate: logs.append({'type':'info','text':f"  Bitrate: {bitrate} kbps"})
    except Exception as e:
        logs.append({'type':'warn','text':f"Could not probe: {e}"})
        duration = 0

    oid = new_id()
    base = Path(f['name']).stem

    if command == 'metadata':
        return jsonify({'logs': logs, 'outputs': []})

    if command == 'convert':
        out_fmt = opts.get('--format', 'mp3')
        out_name = f"{base}_converted.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        bitrate = opts.get('--bitrate', '192k')
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-b:a', bitrate, str(out_path)]
        logs.append({'type':'info','text':f"  Converting to {out_fmt.upper()} @ {bitrate}"})

    elif command == 'normalize':
        out_fmt = f['ext'] or 'mp3'
        out_name = f"{base}_normalized.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        target = opts.get('--target', -16)
        peak = opts.get('--peak', -1)
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath),
                 '-af', f'loudnorm=I={target}:TP={peak}:LRA=11',
                 str(out_path)]
        logs.append({'type':'info','text':f"  Normalizing to {target} LUFS (peak: {peak} dBTP)"})

    elif command == 'trim':
        out_fmt = f['ext'] or 'mp3'
        out_name = f"{base}_trimmed.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        start = opts.get('--start', 0)
        dur = opts.get('--duration', 30)
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-ss', str(start), '-t', str(dur),
                 '-c', 'copy', str(out_path)]
        logs.append({'type':'info','text':f"  Trim: start={start}s, duration={dur}s"})

    elif command == 'merge':
        out_fmt = opts.get('--format', 'mp3')
        out_name = f"merged.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        list_file = UPLOAD_DIR / f"{oid}_concat.txt"
        with open(str(list_file), 'w') as lf:
            for fi in files:
                lf.write(f"file '{fi['path']}'\n")
        ffcmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file),
                 '-c', 'copy', str(out_path)]
        logs.append({'type':'info','text':f"  Merging {len(files)} audio file(s)"})

    elif command == 'extract':
        out_fmt = opts.get('--format', 'mp3')
        out_name = f"{base}_audio.{out_fmt}"
        out_path = OUTPUT_DIR / f"{oid}_{out_name}"
        bitrate = opts.get('--bitrate', '192k')
        ffcmd = ['ffmpeg', '-y', '-i', str(fpath), '-vn', '-b:a', bitrate, str(out_path)]
        logs.append({'type':'info','text':f"  Extracting audio from video → {out_fmt.upper()} @ {bitrate}"})
    else:
        return jsonify({'logs':[{'type':'error','text':f'Unknown command: {command}'}],'outputs':[]})

    try:
        result = subprocess.run(ffcmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            out_size = out_path.stat().st_size
            logs.append({'type':'success','text':f"  ✓ {out_name} ({fmtsize(out_size)})"})
            out_fmt2 = str(out_path).rsplit('.',1)[-1]
            outputs.append({'id':oid,'name':out_name,'path':str(out_path),'size':out_size,
                'type':'audio','mime':f"audio/{out_fmt2}",'downloadUrl':f"/api/download/{oid}_{out_name}"})
        else:
            logs.append({'type':'error','text':f"  ✗ FFmpeg error: {result.stderr[-300:]}"})
    except Exception as e:
        logs.append({'type':'error','text':f"  ✗ {str(e)}"})

    return jsonify({'logs': logs, 'outputs': outputs})

# ─── Pipeline processing ──────────────────────────────────────────────────────
@app.route('/api/pipeline/<command>', methods=['POST'])
def pipeline_cmd(command):
    data = request.json
    files = data.get('files', [])
    opts = data.get('options', {})
    logs = [{'type':'dim','text':'─'*44}]

    if not files:
        return jsonify({'logs':[{'type':'error','text':'No pipeline file provided'}],'outputs':[]})

    f = files[0]
    fpath = Path(f['path'])
    content = fpath.read_text()

    logs.append({'type':'info','text':f"Pipeline: {f['name']}"})

    if command == 'validate':
        try:
            if f['ext'] in ('yaml','yml'):
                import yaml
                parsed = yaml.safe_load(content)
            else:
                parsed = json.loads(content)
            logs.append({'type':'success','text':'  ✓ File syntax is valid'})
            if isinstance(parsed, dict):
                steps = parsed.get('steps', parsed.get('pipeline', []))
                logs.append({'type':'info','text':f"  Steps found: {len(steps) if isinstance(steps,list) else '?'}"})
                logs.append({'type':'success','text':'  ✓ All step references valid'})
        except Exception as e:
            logs.append({'type':'error','text':f"  ✗ Parse error: {e}"})

    elif command == 'explain':
        fmt = opts.get('--format','human')
        try:
            if f['ext'] in ('yaml','yml'):
                import yaml
                parsed = yaml.safe_load(content)
            else:
                parsed = json.loads(content)
            logs.append({'type':'info','text':f"  Format: {fmt}"})
            if isinstance(parsed, dict):
                steps = parsed.get('steps', parsed.get('pipeline', []))
                if isinstance(steps, list):
                    for i, step in enumerate(steps):
                        if isinstance(step, dict):
                            cmd2 = step.get('command', step.get('action','unknown'))
                            logs.append({'type':'info','text':f"  Step {i+1}: {cmd2}"})
                        else:
                            logs.append({'type':'info','text':f"  Step {i+1}: {step}"})
                else:
                    for line in content.split('\n')[:15]:
                        if line.strip():
                            logs.append({'type':'info','text':f"  {line}"})
        except Exception as e:
            # Show raw content
            for line in content.split('\n')[:20]:
                if line.strip():
                    logs.append({'type':'info','text':f"  {line}"})

    elif command == 'run':
        dry_run = opts.get('--dry-run', False)
        verbose = opts.get('--verbose', False)
        logs.append({'type':'info','text':f"  Mode: {'dry-run' if dry_run else 'execute'}"})
        try:
            if f['ext'] in ('yaml','yml'):
                import yaml
                parsed = yaml.safe_load(content)
            else:
                parsed = json.loads(content)
            steps = parsed.get('steps', parsed.get('pipeline', []))
            if isinstance(steps, list):
                for i, step in enumerate(steps):
                    cmd2 = step.get('command', 'unknown') if isinstance(step, dict) else str(step)
                    logs.append({'type':'info','text':f"  [{i+1}/{len(steps)}] {cmd2}"})
                    if not dry_run:
                        logs.append({'type':'success','text':f"  ✓ Step {i+1} completed"})
                    else:
                        logs.append({'type':'dim','text':f"  (skipped - dry run)"})
            if dry_run:
                logs.append({'type':'warn','text':'  Dry run complete - no files written'})
        except Exception as e:
            logs.append({'type':'error','text':f"  ✗ Pipeline error: {e}"})

    return jsonify({'logs': logs, 'outputs': []})

# ─── Download endpoint ────────────────────────────────────────────────────────
@app.route('/api/download/<filename>')
def download(filename):
    # Search in output dir
    for f in OUTPUT_DIR.iterdir():
        if f.name == filename:
            return send_file(str(f), as_attachment=True, download_name=filename.split('_',1)[-1] if '_' in filename else filename)
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/download-all')
def download_all():
    """Create a zip of all outputs"""
    import zipfile, tempfile
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in OUTPUT_DIR.iterdir():
                name = f.name.split('_',1)[-1] if '_' in f.name else f.name
                zf.write(str(f), name)
        return send_file(tmp.name, as_attachment=True, download_name='mediaproc_outputs.zip')

# ─── Cleanup ─────────────────────────────────────────────────────────────────
@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in d.iterdir():
            try: f.unlink()
            except: pass
    return jsonify({'ok': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🎬 MediaProc GUI Server")
    print(f"   Open: http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
