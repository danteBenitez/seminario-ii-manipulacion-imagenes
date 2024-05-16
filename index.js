// @ts-check
// import * as tf from "@tensorflow/tfjs";

const image = document.querySelector("img");
const canvasCropped = document.querySelector("canvas#cropped");
const canvasResized = document.querySelector("canvas#resized-crop");
const canvasResized2 = document.querySelector("canvas#resized-bilinear");
const canvasFlipped = document.querySelector("canvas#flipped");

if (!image) {
  throw new Error("Image not found.");
}

if (!canvasCropped) {
  throw new Error("Canvas not found.");
}

if (!canvasResized) {
  throw new Error("Canvas not found.");
}

if (!canvasResized2) {
  throw new Error("Canvas not found.");
}

const SCALE_FACTOR = 1;
const NUM_BOXES = 1;
const BOX_SIZE = 0.3;

tf.tidy(() => {
  console.log("Leyendo imagen...");
  const imageTensor = tf.browser.fromPixels(image, 4);
  const [ 
    width,
    height,
    depth
  ] = imageTensor.shape;

  const cropped = tf.slice(imageTensor, [0, 40, 0], [265, 300, 3]);
  tf.browser.toPixels(cropped, canvasCropped).then((pixels) => {
    console.log("Imagen recortada.");
  });

  const boxes = tf.tensor2d([[0.5, 0.5, 0.5, 0.5]], [NUM_BOXES, 4]);

  const newSize = [400, 400];
  const scaled = tf.image.resizeNearestNeighbor(cropped, newSize, true);
  tf.browser.toPixels(scaled, canvasResized).then((pixels) => {
    console.log("Imagen redimensionada.");
  });

  const scaled2 = tf.image.resizeBilinear(cropped, newSize, true);
  tf.browser
    .toPixels(scaled2.asType("int32"), canvasResized2)
    .then((pixels) => {
      console.log("Imagen redimensionada.");
    });

  const batch = tf.tensor4d([imageTensor.arraySync()], [1, width, height, depth]);
  const flipped = tf.image.flipLeftRight(batch);
  tf.browser.toPixels(flipped.arraySync()[0], canvasFlipped).then((pixels) => {
    console.log("Imagen volteada.");
  });
});
