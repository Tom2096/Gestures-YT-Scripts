import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam'; 
import ReactPlayer from 'react-player';
import {CSSTransition} from 'react-transition-group';

import * as tf from '@tensorflow/tfjs';
import * as handpose from '@tensorflow-models/handpose';
import {InferenceSession} from 'onnxjs';

import Detect from './components/Detect.js';
import Draw from './components/Draw.js';
import Predict from './components/Predict.js';

import {CircularProgressbar, buildStyles} from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { Icon } from 'semantic-ui-react';
import 'semantic-ui-css/semantic.min.css';

const style = {
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  margin: 'auto',
  width: 640,
  height: 480,
  zIndex: 9,
}

const sensitivity = 15;

const playList = [
  'https://www.youtube.com/watch?v=VY1eFxgRR-k',
  'https://www.youtube.com/watch?v=aJOTlE1K90k',
  'https://www.youtube.com/watch?v=DGzy8FE1Rhk',
  'https://www.youtube.com/watch?v=ycy30LIbq4w',
  'https://www.youtube.com/watch?v=Jqs5EaAaueA',
]

function App() {

  const [tfModel, setTfModel] = useState(null);
  const [onnxModel, setOnnxModel] = useState(null);

  const [showCam, setShowCam] = useState(false);

  const [ts, setTs] = useState(new Date());

  const [queue, setQueue] = useState([]);
  const [action, setAction] = useState(null);
  const [percent, setPercent] = useState(0);

  const [isPlaying, setIsPlaying] = useState(true);
  const [videoIdx, setVideoIdx] = useState(0);

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // INIT ------------------------------------------------------------------------------//

  useEffect(() => {  
    Init();
    const intervalId = setInterval(() => setTs(new Date()), 100);
    return () => clearInterval(intervalId);
  }, [])

  async function Init() {
    const net = await handpose.load();
    setTfModel(net);
    console.log('[-] ... Loaded Model | TensorFlow');

    const sess = new InferenceSession();
    await sess.loadModel('./model.onnx');
    setOnnxModel(sess);
    console.log('[-] ... Loaded Model | Onnx');

    console.log('[-] ... Models Loaded Sucessfully');
  }
  
  // DETECT -----------------------------------------------------------------------------//

  useEffect(() => {
    if ( tfModel && onnxModel) {
      Detect(tfModel, webcamRef, canvasRef).then(
        hand => {
          if (hand && hand.length > 0) {
            const ctx = canvasRef.current.getContext('2d');
            Draw(hand[0], ctx);
            Predict(onnxModel, hand[0]).then(
              prediction => {
                handleQueue(prediction);
              }
            )
          }
          setQueue([]);
          setAction(null);
          setPercent(0);
        }
      )
    }
  }, [ts])

  function handleQueue (prediction) {

    var c_queue = [...queue];

    if (queue.length > sensitivity) {
      c_queue.pop();
    } 
    c_queue.unshift(prediction);
    
    var counter = 0;
    for (var i = 0; i < sensitivity; i++) {
      if ((i === c_queue.length) || (i !== 0 && c_queue[i - 1] !== c_queue[i])) {
        break;
      }
      counter += 1;
    }
    
    if (counter === sensitivity) {
      if (action === 'play') {
        setIsPlaying(true);
      }
      if (action === 'pause') {
        setIsPlaying(false);
      }
      if (action === 'previous') {
        previousVideo();
      }
      if (action === 'next') {
        nextVideo();
      }
      
      console.log('[-] ... Action Completed | ' + action);
      
      c_queue = [];  
    }
    
    setQueue(c_queue);
    setAction(c_queue[0]);
    setPercent(Math.round((counter / sensitivity) * 100));
  }

  // PLAYLIST ----------------------------------------------------------------------------//

  function nextVideo() {
    const c_idx = videoIdx;
    if (c_idx === playList.length - 1) {
      setVideoIdx(0);
    } else {
      setVideoIdx(c_idx + 1);
    }
  }

  function previousVideo() {
    const c_idx = videoIdx;
    if (c_idx === 0) {
      setVideoIdx(playList.length - 1);
    } else {
      setVideoIdx(c_idx - 1);
    }
  }

  // RENDER ------------------------------------------------------------------------------//

  return (
    <div className="app">

      <Icon
        onMouseEnter={() => {setShowCam(true)}}
        onMouseLeave={() => {setShowCam(false)}} 
        name='record' 
      />

      <Webcam
        className={'webcam ' + (showCam ? 'active' : '')}
        ref={webcamRef}
        style={style}
      />

      <canvas 
        ref={canvasRef} 
        style={style}>
      </canvas>

      <div className={'overlay ' + (showCam ? 'active' : '')}>
      </div>

      <div className='content'>
        <div className='video-player'>
          <div className='player-wrapper'>
            <ReactPlayer
              playing={isPlaying}
              onEnded={nextVideo}
              onPause={() => setIsPlaying(false)}
              className='react-player'
              url={playList[videoIdx]}
              width='100%'
              height='100%'
            />
            <CSSTransition
              in={action !== null}
              appear={false}
              unmountOnExit={true}
              timeout={400}
              classNames={'fade'}
            >
              <CircularProgressbar
                background
                value={percent}
                text={action}
                strokeWidth={5}
                styles={buildStyles({
                  pathTransitionDuration: 0.1,
                  backgroundColor: 'rgba(0 ,0, 0, 0.8)',
                  pathColor: 'white',
                  trailColor: 'black',
                  textColor: 'white',
                })}
              />
            </CSSTransition>
            
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
