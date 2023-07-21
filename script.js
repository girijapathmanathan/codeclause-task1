window.addEventListener("load", async function() {
    const image = document.getElementById("image");
    const canvas = document.getElementById("foreground");
    const ctx = canvas.getContext("2d");
    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;

    
    const model = await deeplab.load();


    async function removeBackground() {

        const tfImage = tf.browser.fromPixels(image);
        const tfPreprocessed = tf.image.resizeBilinear(tfImage, [canvas.height, canvas.width]);
        const tfNormalized = tf.div(tfPreprocessed, tf.scalar(255));
        const tfBatched = tf.expandDims(tfNormalized);
        const tfSegmentation = await model.segment(tfBatched);
        const tfData = tfSegmentation.dataSync();
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        for (let i = 0; i < tfData.length; i++) {
            imageData.data[i * 4 + 3] = tfData[i] === 0 ? 0 : 255; 
        }
        ctx.putImageData(imageData, 0, 0);
        tfImage.dispose();
        tfPreprocessed.dispose();
        tfNormalized.dispose();
        tfBatched.dispose();
        tfSegmentation.dispose();
    }
    image.onload = function() {
        removeBackground();
    };
});
