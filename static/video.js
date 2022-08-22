var base64Canvas;
var video = document.getElementById("video");
var canvas = document.getElementById("framecanvas");
var interval = 1000;  // 1000 = 1 second, 3000 = 3 seconds


//Setup socket con variabili
const sio = io()
sio.connect()

//Imposto frame da analizzare al secondo
function setFrameSentPerSecond(n){
    interval = interval / n
    loadcanvas()
}


function loadcanvas(){

    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    base64Canvas = canvas.toDataURL("image/jpeg").split(';base64,')[1];
    sio.emit('frame-video', base64Canvas)

    setTimeout(loadcanvas, interval);
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

