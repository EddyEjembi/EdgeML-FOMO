
## **Introduction**
Read the full article on [Here](https://medium.com/@eddyejembi/tiny-ml-bringing-ai-to-the-edge-9d89d7d779c4)

In this project, I built a lightweight image classification model, drawing inspiration from [Edge Impulse’s](https://edgeimpulse.com/) [FOMO](https://www.edgeimpulse.com/blog/announcing-fomo-faster-objects-more-objects/) by modifying the [**MobileNetV2**](https://arxiv.org/abs/1801.04381) architecture, optimizing it for the Edge, and converting it into **ONNX and TensorFlow Lite Micro** formats for deployment.

This article explains the methodology, results, and potential applications of this approach.

## **Why FOMO and MobileNetV2?**

[**MobileNetV2**](https://arxiv.org/abs/1801.04381) is a well-known lightweight convolutional neural network (CNN) designed to perform well on mobile devices. It uses **depth-wise separable convolutions, inverted residuals, and linear bottlenecks** to improve efficiency.

However, even though it’s optimized, deploying it directly on microcontrollers can still be challenging. My target microcontrollers is the [**ESP32-CAM**](https://docs.sunfounder.com/projects/galaxy-rvr/en/latest/hardware/cpn_esp_32_cam.html), and the model is too large for it’s specifications:

📌 **520 KB RAM + 4MB PSRAM** (too little for standard MobileNetV2).

📌 **4MB flash storage**.

📌 **Limited compute power** (lacks a dedicated AI accelerator).

To make the model even smaller, I applied the key idea behind the **FOMO** architecture, removing unnecessary complexity while retaining essential features. Instead of using full MobileNetV2, I modified its architecture to focus on lightweight feature extraction. The steps taken in implementing this were;

- **Cut MobileNetV2 at the 7th layer**, significantly reducing the model size while retaining useful feature extraction capabilities.
- **Fine-tuned it on a fruit classification dataset** to improve its performance.

The goal is to create an highly efficient model for Edge ML that can run on low-powered hardware.

## **Implementation: Training and Optimization**

### **1. Modifying MobileNetV2 for Edge Deployment**

Starting with MobileNetV2, a lightweight CNN. Running the full model on an ESP32-CAM can be impractical due to RAM and compute limitations. The idea was to Cut the Model at the 7th Layer. MobileNetV2 consists of multiple convolutional layers, but running all of them would be too heavy for a microcontroller. By removing the deeper layers and keeping only the first 7, I retained the feature extraction capability while significantly reducing computational cost.

```python
self.features = nn.Sequential(*list(model_head.features.children())[:7])
```

This modification reduces the model size and inference time, making it more suitable for edge deployment.

---

### **2. Adding a Lightweight Detection Head**

Removing the classification layers of MobileNetV2 requires to add a new classification head for feature aggregation and classification. I used a 1x1 Convolutional Layer & ReLU Activation. A **1x1 convolution** is computationally efficient and helps reduce the number of parameters while maintaining feature depth. The **ReLU activation** introduces non-linearity, improving learning, and then there is a **Dropout** to prevent the network from being too dependent on any particular feature.

```python
self.head = nn.Sequential(
    nn.Conv2d(last_channel, 32, kernel_size=1, stride=1),
    nn.ReLU(),
    nn.Dropout(0.5)
)
```

This step allows the model to extract more compact, useful features before classification.

---

### **3. Using Global Average Pooling for Efficient Classification**

The classification head now outputs spatial feature maps, so I needed to convert them into a single vector for final classification by Applying [**Adaptive Average Pooling**](https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work). This reduces dimensionality while preserving important features, making it ideal for lightweight models.

```python
self.pool = nn.AdaptiveAvgPool2d((1, 1))
```

---

### **4. Adding the Final Classifier**

Since this is a classification task, a final logits layer was needed to map the processed features to class predictions. Instead of a fully connected layer (which is memory-intensive), a [**1x1 convolution**](https://stackoverflow.com/a/39367644) acts as a classifier while keeping the model lightweight.

```python
self.logits = nn.Conv2d(32, num_classes, kernel_size=1, stride=1)
```

---

### **5. Fine-Tuning and Training**

With the architecture modified, I trained the model on a [**fruit classification dataset**](https://www.kaggle.com/datasets/philschmid/tiny-fruit-object-detection) for 25 epochs and achieved the following result:

- 📌 **Test Loss:** 0.7547
- 📌 **Accuracy:** 76.67%

### **6. Model Conversion for Microcontroller Deployment**

After training, I needed to convert the model to formats suitable for **low-power inferencing.** To do so**,** I followed two steps:

- **Step 1: Convert to ONNX** – [ONNX](https://onnx.ai/) allows interoperability between frameworks, making it easier to optimize the model further.
- **Step 2: Convert ONNX to TensorFlow Lite Micro (TFLite-Micro)** – [TFLite-Micro](https://ai.google.dev/edge/litert/microcontrollers/overview) comes with it’s own optimization techniques which applies quantization, reducing the model size and making it lightweight enough for microcontrollers.

Now, the model is optimized and ready for deployment! 🚀

## **Final Thoughts**

In **continuation**, I will deploy this model as a C++ library and inference it on an ESP32-CAM for real-world testing. Stay tuned! 🚀

Read the full article on [Medium](https://medium.com/@eddyejembi/tiny-ml-bringing-ai-to-the-edge-9d89d7d779c4)

💬 Thoughts or feedback? [Let’s discuss](mailto:eddyejembi2018@gmail.com)

### References

https://www.edgeimpulse.com/blog/announcing-fomo-faster-objects-more-objects/

https://www.edgeimpulse.com/blog/fomo-self-attention/

https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices

https://arxiv.org/abs/1801.04381

