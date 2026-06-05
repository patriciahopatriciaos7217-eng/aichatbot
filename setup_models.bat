@echo off
echo Pulling required Ollama models...
echo.

echo Pulling phi3...
ollama pull phi3

echo Pulling llama3.2...
ollama pull llama3.2

echo Pulling nomic-embed-text (for embeddings)...
ollama pull nomic-embed-text

echo Pulling mxbai-embed-large (for embeddings)...
ollama pull mxbai-embed-large


echo ✅ All models pulled successfully!
ollama list