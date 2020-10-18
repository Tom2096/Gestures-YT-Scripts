import {Tensor, InferenceSession} from 'onnxjs';

const labels_dict = {
    0: 'play',
    1: 'pause',
    2: 'previous',
    3: 'next',
    4: 'ok',
}

function normalize(landmarks) {

    if (landmarks.length === 0) {
        return null;
    }

    var x_max, x_min, y_max, y_min;

    x_max = x_min = landmarks[0][0];
    y_max = y_min = landmarks[0][1];

    landmarks.forEach(landmark => {
        x_max = Math.max(x_max, landmark[0]);
        x_min = Math.min(x_min, landmark[0]);

        y_max = Math.max(y_max, landmark[1]);
        y_min = Math.min(y_min, landmark[1]);
    })

    const w = x_max - x_min;
    const h = y_max - y_min;

    const data = [];

    landmarks.forEach(landmark => {
        const x = (landmark[0] - x_min) / w;
        const y = (landmark[1] - y_min) / h;
        data.push(x, y);
    })
    
    return data;
}

function maxIdx(array) {
    if (array.length === 0) {
        return -1;
    }

    var max = array[0];
    var idx = 0;

    for (var i = 0; i < array.length; i++) {
        if (array[i] > max) {
            max = array[i];
            idx = i;
        }
    }

    return idx;
}

async function Predict(model, prediction) {

    const landmarks = prediction.landmarks;

    const input = [new Tensor (normalize(landmarks), 'float32', [1, 21, 2])];
    const outputMap = await model.run(input);
    const outputTensor = outputMap.values().next().value;

    const label = labels_dict[maxIdx(outputTensor.data)];

    return label; 
}

export default Predict;
