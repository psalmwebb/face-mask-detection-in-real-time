
let xhr = new XMLHttpRequest()


xhr.open('get','/stream-cam')

xhr.onload =()=>{
    console.log(xhr.responseText)
}


xhr.send()
