#!/bin/bash

# Script to restart Ollama with proper GPU settings

echo "Stopping Ollama..."
sudo snap stop ollama

echo "Setting GPU environment variables..."
export OLLAMA_GPU_LAYERS=-1
export OLLAMA_GPU_MEMORY_FRACTION=0.9

echo "Starting Ollama with GPU settings..."
sudo snap start ollama

echo "Waiting for Ollama to start..."
sleep 5

echo "Checking Ollama status..."
ollama ps

echo "Testing GPU usage with a simple call..."
ollama run gemma3:12b-it-q8_0 "Test GPU usage" --gpu-layers -1



