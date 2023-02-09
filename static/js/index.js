//index.js
var colors = [
    '#F2B705',
    '#F25C05',
    '#0388A6',
    '#0E5929',
    '#F272A1',
]

var container = document.getElementById('container');

var text = document.getElementById('value');	//Scroll Value
var color = document.getElementById('color');	//Color Code

//Wheel Event
container.onwheel = changeBgColor;

var colorIndex = 0;
var scrollValue = 0;	//get mouse wheel value

function changeBgColor(e) {
    scrollValue += e.deltaY * 0.01;

    console.log(scrollValue);

    //위로 Scroll
    if (scrollValue > 10) {
        colorIndex += 1;
        if (colorIndex > colors.length-1) colorIndex = 0;
        color.textContent = colors[colorIndex];
        container.style.backgroundColor = colors[colorIndex];	//배경색 변경
        scrollValue = 0;	//스크롤 값 초기화
    }

    //아래로 Scroll
    if (scrollValue < -10) {
        colorIndex -= 1;
        if (colorIndex < 0) colorIndex = colors.length-1;
        color.textContent = colors[colorIndex];
        container.style.backgroundColor = colors[colorIndex];	//배경색 변경
        scrollValue = 0;	//스크롤 값 초기화
    }
    e.preventDefault(); // disable the actual scrolling
}