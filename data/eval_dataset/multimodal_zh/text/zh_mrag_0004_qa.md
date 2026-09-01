# RAG项目架构图

## 图片主题
求职助手 RAG 架构

## OCR文本
文本简历/JD -> 语义切块 -> Chroma; 图片简历/JD -> GLM-V OCR/语义抽取 -> Chroma; 检索 -> Rerank -> Prompt -> GLM 生成

## 问题
架构图中图片资料如何进入统一检索链路？

## 标准答案
图片先经过 GLM-V OCR/语义抽取，再写入 Chroma，与文本统一检索。

## 检索关键词
GLM-V, OCR, 语义抽取, Chroma, 统一检索
