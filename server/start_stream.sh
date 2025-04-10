#!/bin/bash
export $(grep -v '^#' .env | xargs)
mkdir -p ./static/live

ffmpeg -rtsp_transport tcp -analyzeduration 10000000 -probesize 5000000 \
  -i "rtsp://admin:${CAMERA_PASSWORD}@${IP_ADDRESS}:${PORT}/Streaming/Channels/${CAMERA_ID}" \
  -vf "scale=1280:720" \
  -c:v libx264 -preset veryfast -g 48 -sc_threshold 0 \
  -f hls -hls_time 2 -hls_list_size 5 -hls_flags delete_segments+append_list \
  ./static/live/stream.m3u8
