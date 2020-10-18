function Draw(prediction, ctx) {

    const landmarks = prediction.landmarks;

    landmarks.forEach(landmark => {
        const x = landmark[0];
        const y = landmark[1];
        
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 3 * Math.PI);
        ctx.fillStyle = "black";
        ctx.fill();

    });
}

export default Draw;