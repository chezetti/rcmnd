"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { toast } from "@/components/ui/use-toast";
import { Toaster } from "@/components/ui/toaster";
import Uppy from "@uppy/core";
import "@uppy/core/dist/style.min.css";
import "@uppy/dashboard/dist/style.min.css";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import Header from "@/components/layout/header";
import Footer from "@/components/layout/footer";
import apiService from "@/lib/api";
import { FileUploader } from "@/components/ui/file-uploader";

export default function UploadPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [uppyInstance] = useState(
    () =>
      new Uppy({
        id: "upload-uppy",
        restrictions: {
          maxNumberOfFiles: 1,
          allowedFileTypes: [".jpg", ".jpeg", ".png", ".webp"],
        },
        autoProceed: false,
      })
  );

  // Create react-query mutation
  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await apiService.uploadNFT(formData);
      return response.data;
    },
    onSuccess: (data) => {
      toast({
        title: "Upload successful!",
        description: `Your NFT has been added with ID: ${data.uuid}`,
      });

      // Reset form
      setName("");
      setDescription("");
      setTags("");

      // Сброс файлов в Uppy
      if (uppyInstance && uppyInstance.getFiles().length > 0) {
        uppyInstance.getFiles().forEach((file) => {
          uppyInstance.removeFile(file.id);
        });
      }

      // Redirect to the item page after a short delay
      setTimeout(() => {
        router.push(`/item/${data.uuid}`);
      }, 2000);
    },
    onError: (error) => {
      toast({
        title: "Upload failed",
        description:
          error instanceof Error ? error.message : "An unknown error occurred",
        variant: "destructive",
      });
    },
  });

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate form
    if (!name || !description || uppyInstance.getFiles().length === 0) {
      toast({
        title: "Validation error",
        description: "Please provide a name, description, and image.",
        variant: "destructive",
      });
      return;
    }

    // Prepare form data
    const formData = new FormData();
    const file = uppyInstance.getFiles()[0];

    formData.append("image", file.data);
    formData.append("name", name);
    formData.append("description", description);
    if (tags) {
      formData.append("tags", tags);
    }

    // Execute upload
    uploadMutation.mutate(formData);
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <Toaster />
      <main className="flex-1 container py-10">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-4">Upload NFT</h1>
          <p className="text-muted-foreground">
            Add your NFT to the collection.
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
              {/* Image upload */}
              <div className="p-6 bg-card/30 backdrop-blur-sm rounded-lg">
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                  <span className="bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                    Upload NFT Image
                  </span>
                </h2>
                <FileUploader
                  uppy={uppyInstance}
                  height={300}
                  note="Only JPG, PNG and WebP images are allowed"
                />
              </div>

              {/* NFT details */}
              <div className="border rounded-lg p-6 bg-card h-full">
                <h2 className="text-xl font-semibold mb-4">NFT Details</h2>

                <div className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="name" className="form-field-label">
                      Name
                    </Label>
                    <Input
                      id="name"
                      placeholder="NFT Name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="description" className="form-field-label">
                      Description
                    </Label>
                    <Textarea
                      id="description"
                      placeholder="Describe your NFT..."
                      className="min-h-[120px]"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="tags" className="form-field-label">
                      Tags (comma separated)
                    </Label>
                    <Input
                      id="tags"
                      placeholder="art, abstract, cyberpunk, etc."
                      value={tags}
                      onChange={(e) => setTags(e.target.value)}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Submit button */}
            <div className="flex justify-center w-full mt-10">
              <div className="upload-nft-btn w-full max-w-md mx-auto">
                <Button
                  type="submit"
                  size="lg"
                  className="w-full"
                  disabled={uploadMutation.isPending}
                >
                  {uploadMutation.isPending ? "Uploading..." : "Upload NFT"}
                </Button>
              </div>
            </div>
          </form>
        </motion.div>
      </main>
      <Footer />
    </div>
  );
}
