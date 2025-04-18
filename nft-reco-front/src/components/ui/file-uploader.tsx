"use client";

import React from "react";
import { Dashboard } from "@uppy/react";
import Uppy from "@uppy/core";
import "@uppy/core/dist/style.min.css";
import "@uppy/dashboard/dist/style.min.css";
import { cn } from "@/lib/utils";

// Импортируем пользовательские стили для загрузчика файлов
import "./file-uploader-styles.css";

interface FileUploaderProps {
  className?: string;
  onFilesSelected?: (files: File[]) => void;
  allowMultiple?: boolean;
  maxNumberOfFiles?: number;
  maxFileSize?: number;
  allowedFileTypes?: string[];
  uppy?: Uppy;
  height?: number;
  note?: string;
}

export function FileUploader({
  className,
  onFilesSelected,
  allowMultiple = false,
  maxNumberOfFiles = 1,
  maxFileSize = 10 * 1024 * 1024, // 10MB по умолчанию
  allowedFileTypes = ["image/*"],
  uppy,
  height = 300,
  note = "Only JPG, PNG and WebP images are allowed",
}: FileUploaderProps) {
  const uppyInstance = React.useMemo(() => {
    if (uppy) return uppy;

    return new Uppy({
      id: "uppy-file-uploader",
      autoProceed: true,
      restrictions: {
        maxNumberOfFiles: allowMultiple ? maxNumberOfFiles : 1,
        maxFileSize,
        allowedFileTypes,
      },
    });
  }, [uppy, maxNumberOfFiles, maxFileSize, allowedFileTypes, allowMultiple]);

  React.useEffect(() => {
    if (onFilesSelected) {
      uppyInstance.on("complete", (result) => {
        if (result.successful) {
          const files = result.successful.map((file) => file.data as File);
          onFilesSelected(files);
        }
      });
    }

    return () => {
      if (!uppy) {
        uppyInstance.cancelAll();
      }
    };
  }, [uppyInstance, onFilesSelected, uppy]);

  return (
    <div className={cn("file-uploader-container", className)}>
      <div className="gradient-border">
        <Dashboard
          uppy={uppyInstance}
          proudlyDisplayPoweredByUppy={false}
          showProgressDetails
          height={height}
          width="100%"
          note={note}
          className="custom-uppy-dashboard uppy-centered-text"
          metaFields={[
            { id: "name", name: "Name", placeholder: "Artwork name" },
          ]}
        />
      </div>
    </div>
  );
}
