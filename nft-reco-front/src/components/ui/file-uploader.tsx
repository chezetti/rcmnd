"use client";

import React from "react";
import { Dashboard } from "@uppy/react";
import Uppy from "@uppy/core";
import "@uppy/core/dist/style.min.css";
import "@uppy/dashboard/dist/style.min.css";
import { cn } from "@/lib/utils";

// Создаем пользовательские стили для загрузчика файлов
import "./file-uploader-styles.css";

interface FileUploaderProps {
  uppy: Uppy;
  height?: number;
  width?: string | number;
  note?: string;
  className?: string;
}

export function FileUploader({
  uppy,
  height = 300,
  width = "100%",
  note = "Only JPG, PNG and WebP images are allowed",
  className,
}: FileUploaderProps) {
  return (
    <div className={cn("file-uploader-container", className)}>
      <div className="gradient-border">
        <Dashboard
          uppy={uppy}
          proudlyDisplayPoweredByUppy={false}
          showProgressDetails
          height={height}
          width={width}
          note={note}
          className="custom-uppy-dashboard"
        />
      </div>
    </div>
  );
}
