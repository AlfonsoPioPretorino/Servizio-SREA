//Setup socket
const sio = io()
sio.connect()

//Setup della webcam
var video = document.getElementById("video");
var canvas = document.getElementById("framecanvas");
var mediaDevices = navigator.mediaDevices;


var interval = 1000;  // 1000 = 1 second, 3000 = 3 seconds

mediaDevices.getUserMedia({
    video: true,
    audio: false,
})
.then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadedmetadata", () => {
    video.play();
    });
})
.catch(alert);

//Loop d'invio frames
setTimeout(loadcanvas, interval);
function loadcanvas(){
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    base64Canvas = canvas.toDataURL("image/jpeg").split(';base64,')[1];
    sio.emit('frame-webcam', base64Canvas)
    setTimeout(loadcanvas, interval);
}

//Imposto frame da analizzare al secondo
function setFrameSentPerSecond(n){
    interval = interval / n
    loadwebcam()
}

//Listeners e eventi di callback
sio.on('connect', () => {
    console.log('Successfully connected!');
  });
  
  sio.on('disconnect', () => {
      console.log('Successfully disconnected!');
  });
  
  sio.on('results', (data)=>{
      $('#stats').children().remove();
      let header = ["Emozione rilevata", "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness"]
      console.log(data)
      let datas = data.split("|")
      if (datas.length == 1){
          $("#stats").append('<h5 id="update"><b>'+datas[0]+'</b></h5>');
      }
      else{
          $("#stats").append('<div id ="EmoRil"><h5><b>Emozione Rilevata:</b></h5></div>')
          $("#stats").append('<div id ="RetVals"><h5><b>Valori restituiti:</b></h5></div> ')
          for(let i = 0; i<datas.length-1; i++){
              if(i == 0){
                  $("#EmoRil").append('<p style="text-align: left ;">'+header[i]+': '+datas[i]+'</p>');
              }
              else{
                  $("#RetVals").append('<p style="text-align: left ;">'+header[i]+': '+datas[i]+'%</p>');
              }
          }
      }
  
  });


//Disconnessione socket quando la pagina viene distrutta
window.onbeforeunload = function(){
    sio.disconnect()
}