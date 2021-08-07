var userAgent = window.navigator.userAgent.toLowerCase();
console.log(userAgent);
var norunAI = [ "android", "iphone", "ipod", "ipad", "windows phone", "mqqbrowser" ,"msie","trident/7.0"];
var norunFlag = false;
for(var i=0;i<norunAI.length;i++){
	if(userAgent.indexOf(norunAI[i]) > -1){
		norunFlag = true;
		break;
	}
}
if(!window.WebGLRenderingContext){
	norunFlag = true;
}
if(!norunFlag){
    var AIFadeFlag = false;
    /* renderTip */
    function renderTip(template, context) {
        var tokenReg = /(\\)?\{([^\{\}\\]+)(\\)?\}/g;
        return template.replace(tokenReg, function (word, slash1, token, slash2) {
            if (slash1 || slash2) {
                return word.replace('\\', '');
            }
            var variables = token.replace(/\s/g, '').split('.');
            var currentObject = context;
            var i, length, variable;
            for (i = 0, length = variables.length; i < length; ++i) {
                variable = variables[i];
                currentObject = currentObject[variable];
                if (currentObject === undefined || currentObject === null) return '';
            }
            return currentObject;
        });
    }
    String.prototype.renderTip = function (context) {
        return renderTip(this, context);
    };
    /* Console */
    var open_console = function(){};
    console.log(open_console); // 控制台log函数时会调用toString方法
    open_console.toString = function() {
        showMessage('哈哈，你打开了控制台，是想要看看我的秘密吗？', 5000);
        return '';
    }
    /* Copy */
    $(document).on('copy', function (){
		showMessage('你都复制了些什么呀，转载要记得加上出处哦~~', 5000);
	});
	/* Click and MouseOver MessageInit */
	function initTips(){
        $.ajax({
            cache: true,
            url: `${message_Path}message.json`,
            dataType: "json",
            success: function (result){
                $.each(result.mouseover, function (index, tips){
                    $(tips.selector).mouseover(function (){
                        if ($('#live_statu_val').val() == "0") {
                            var text = tips.text;
                            if(Array.isArray(tips.text)) text = tips.text[Math.floor(Math.random() * tips.text.length + 1)-1];
                            text = text.renderTip({text: $(this).text()});
                            showMessage(text, 3000);
                        }
                    });
                });
                $.each(result.click, function (index, tips){
                    $(tips.selector).click(function (){
                        if ($('#live_statu_val').val() == "0") {
                            var text = tips.text;
                            if(Array.isArray(tips.text)) text = tips.text[Math.floor(Math.random() * tips.text.length + 1)-1];
                            text = text.renderTip({text: $(this).text()});
                            showMessage(text, 3000);
                        }
                    });
                });
            }
        });
    }
    initTips();
    /* StartMessage and TimeMessage */
    var text;
    console.log("document.referrer:", document.referrer) // 来访页面的URL
    console.log("window.location.href", window.location.href) // 当前页面的URL
    if(document.referrer !== ''){
        var referrer = document.createElement('a');
        referrer.href = document.referrer;
        text = '嗨！来自 <span style="color:#0099cc;">' + referrer.hostname + '</span> 的朋友！';
        var domain = referrer.hostname.split('.')[1];
        if (domain == 'baidu') {
            text = '嗨！ 来自 百度搜索 的朋友！<br>欢迎访问<span style="color:#0099cc;">「 ' + document.title.split(' - ')[0] + ' 」</span>';
        }else if (domain == 'so') {
            text = '嗨！ 来自 360搜索 的朋友！<br>欢迎访问<span style="color:#0099cc;">「 ' + document.title.split(' - ')[0] + ' 」</span>';
        }else if (domain == 'google') {
            text = '嗨！ 来自 谷歌搜索 的朋友！<br>欢迎访问<span style="color:#0099cc;">「 ' + document.title.split(' - ')[0] + ' 」</span>';
        }
    } else {
        if (window.location.href == `${home_Path}`) { //主页URL判断，需要斜杠结尾
            var now = (new Date()).getHours();
            if (now > 23 || now <= 5) {
                text = '你是夜猫子呀？这么晚还不睡觉，明天起的来嘛？';
            } else if (now > 5 && now <= 7) {
                text = '早上好！一日之计在于晨，美好的一天就要开始了！';
            } else if (now > 7 && now <= 11) {
                text = '上午好！工作顺利嘛，不要久坐，多起来走动走动哦！';
            } else if (now > 11 && now <= 14) {
                text = '中午了，工作了一个上午，现在是午餐时间！';
            } else if (now > 14 && now <= 17) {
                text = '午后很容易犯困呢，今天的运动目标完成了吗？';
            } else if (now > 17 && now <= 19) {
                text = '傍晚了！窗外夕阳的景色很美丽呢，最美不过夕阳红~~';
            } else if (now > 19 && now <= 21) {
                text = '晚上好，今天过得怎么样？';
            } else if (now > 21 && now <= 23) {
                text = '已经这么晚了呀，早点休息吧，晚安~~';
            } else {
                text = '嗨~ 快来逗我玩吧！';
            }
        } else {
            text = '欢迎<span style="color:#0099cc;">「 ' + document.title.split(' - ')[0] + ' 」</span>';
        }
    }
    showMessage(text, 12000);
    /* Show hitokoto Message*/
    interval_func = function () {
        $.ajax({
            url:"https://v1.hitokoto.cn/",
            type:"get",
            dataType:"json",
            success: function(data) {
                showMessage(data.hitokoto + "----" + data.from, 6000);
            }
        });
    };
    var hitokoto = setInterval(interval_func, 25000);
    /* initLive2D */
    function initLive2D() {
        /* 隐藏开启面板娘 */
        $('#hideButton').on('click', function(){
            console.log("hide")
			if(AIFadeFlag){
				return false;
			} else{
				AIFadeFlag = true;
				localStorage.setItem("live2dhidden", "0");
				$('#landlord').fadeOut(200);
				$('#open_live2d').delay(200).fadeIn(200);
				setTimeout(function(){
					AIFadeFlag = false;
				},300);
			}
		});
		$('#open_live2d').on('click', function(){
			if(AIFadeFlag){
				return false;
			}else{
				AIFadeFlag = true;
				localStorage.setItem("live2dhidden", "1");
				$('#open_live2d').fadeOut(200);
				$('#landlord').delay(200).fadeIn(200);
				setTimeout(function(){
					AIFadeFlag = false;
				},300);
			}
		});
		/* AI聊天 */
		$('#showInfoBtn').on('click',function(){
			var live_statu = $('#live_statu_val').val();
			if(live_statu=="0"){
				return
			} else {
	            hitokoto = setInterval(interval_func, 25000);
				$('#live_statu_val').val("0");
				$('.live_talk_input_body').fadeOut(500);
				$('#showTalkBtn').show();
				$('#showInfoBtn').hide();
			}
		});
		$('#showTalkBtn').on('click',function(){
			var live_statu = $('#live_statu_val').val();
			if(live_statu=="1"){
				return
			}else{
			    clearInterval(hitokoto);
				$('#live_statu_val').val("1");
				$('.live_talk_input_body').fadeIn(500);
				$('#showTalkBtn').hide();
				$('#showInfoBtn').show();
			}
		});
		$('#talk_send').on('click',function(){
			var info_ = $('#AIuserText').val();
			var userid_ = $('#AIuserName').val();
			if(info_ == "" ){
				showMessage('写点什么吧！',3000);
				return;
			}
			if(userid_ == ""){
				showMessage('聊之前请告诉我你的名字吧！',3000);
				return;
			}
			$.ajax({
				type: 'POST',
				url: `http://${home_Path}/talk/`,
				data: {'id':userid_, 'info':info_},
				success: function(data) {
				    console.log(data);
                    showMessage(data, 3000);
				}
			});
			$('#AIuserText').val("");
		});
		$("body").keydown(function() {
            if (event.keyCode == "13" && $('#live_statu_val').val() == "1") {//keyCode=13是回车键
                $('#talk_send').click();
            }
        });

		/* 换装 */
		var current_suit = 'seifuku.model.json';
		$('#huanzhuangButton').on('click', function() {
		    model_names = ['ryoufuku.model.json', 'seifuku.model.json', 'shifuku.model.json'];
		    model_select = model_names[Math.floor(Math.random() * 3 + 1)-1];
		    while(model_select == current_suit) {
		        model_select = model_names[Math.floor(Math.random() * 3 + 1)-1];
		    };
		    loadlive2d("live2d", `${static_Path}live2d/model/mashiro/${model_select}`);
		    current_suit = model_select;
		});
		/* 页面抖动,彩虹色变化 */
		$('#youduButton').on('click',function(){
			if($('#youduButton').hasClass('doudong')){
				var typeIs = $('#youduButton').attr('data-type');
				$('#youduButton').removeClass('doudong');
				$('body').removeClass(typeIs);
				$('#youduButton').attr('data-type','');
			}else{
				var duType = $('#duType').val();
				var duArr = duType.split(",");
				var dataType = duArr[Math.floor(Math.random() * duArr.length)];
				$('#youduButton').addClass('doudong');
				$('#youduButton').attr('data-type',dataType);
				$('body').addClass(dataType);
			}
		});
		/* 音乐 */
		// ajax post 403 问题
        function getCookie(name) {
            var cookieValue = null;
            if (document.cookie && document.cookie != '') {
                var cookies = document.cookie.split(';');
                for (var i = 0; i < cookies.length; i++) {
                    var cookie = jQuery.trim(cookies[i]);
                    // Does this cookie string begin with the name we want?
                    if (cookie.substring(0, name.length + 1) == (name + '=')) {
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }
                }
            }
            return cookieValue;
        }
        function csrfSafeMethod(method) {
            // these HTTP methods do not require CSRF protection
            return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
        }
        $.ajaxSetup({
            beforeSend: function(xhr, settings) {
            var csrftoken = getCookie('csrftoken');
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                    xhr.setRequestHeader("X-CSRFToken", csrftoken);
                }
            }
        });
        const ap = new APlayer({
            container: document.getElementById('aplayer'),
            fixed: true,
            listMaxHeight: '230px',  // 设置列表最大高度，需要加px
            lrcType: 3
        });
        var isLoaded = false;
        $('#musicButton').on('click',function(){
            if($('#aplayer').css("display") == 'none') {
                $('#aplayer').show()
                showMessage(`打开了播放器`, 5000);
                if (isLoaded == false) {
                    // 获取音乐列表
                    var music_names = ["EGOIST - Departures~あなたにおくるアイの歌~.mp3",
                    "Jayesslee - Officially Missing You.mp3",
                    "JNique Nicole - Weight of the World (English Version).mp3",
                    "Mili - Nine Point Eight.mp3",
                    "Mili - Past the Stargazing Season.mp3",
                    "Mili - world.execute (me) ;.mp3",
                    "Riin - Alice good night.mp3",
                    "ryo,初音ミク - ODDS&ENDS.mp3",
                    "塞壬唱片-MSR、DJ OKAWARI、二宮愛 - Speed of Light.mp3",
                    "郑成河 - River_Flows_In_You_(Yiruma).mp3"];
                    for (i in music_names) {
                        if (music_names[i].split('.').pop() == 'mp3') {
                            music_name = music_names[i];
                            console.log('music',music_name);
                            split_music_name = music_name.split('.');
                            suffix = split_music_name.pop();
                            name = split_music_name.join(".");
                            ap.list.add([{
                                name: name,
                                artist: name.split('-')[0],
                                url: `${static_Path}music/${music_name}`,
                                cover: `${static_Path}music/${name}.jpg`,
                                lrc: `${static_Path}music/${name}.lrc`,
                            }]);
                        }
                    }
                    ap.play();
                    $('#musicButton').addClass('play');
                    isLoaded = true;
                } else {
                    ap.play();
                }
                $("body").keydown(function() {
                    if (event.keyCode == "32" && $('#aplayer').css("display") != 'none') {//keyCode=32是空格键
                        ap.toggle();
                    }
                });
            } else {
                $('#aplayer').hide()
                showMessage(`关闭了播放器`, 5000);
                ap.pause();
                $('#musicButton').removeClass('play');
            }
//            if($('#musicButton').hasClass('play')){
//                $('#live2d_bgm')[0].pause();
//                $('#musicButton').removeClass('play');
//            }else{
//                $('#live2d_bgm')[0].play();
//                $('#musicButton').addClass('play');
//                showMessage(`正在播放${music_name}`, 5000);
//            }
//            $('#live2d_bgm').bind('ended',function () {
//               $('#musicButton').removeClass('play');
//               music_name = music_list.responseJSON[Math.floor(Math.random() * music_list_length + 1)-1];
//               $('#live2d_bgm').attr('src', `${static_Path}music/${music_name}`);
//            });
        });
		/* 移动 */
//		$(function() {
//            $( "#landlord" ).draggable({
//              scroll: false
//            });
//        });
    }
    initLive2D();
};

function showMessage(text, timeout){
    if(Array.isArray(text)) text = text[Math.floor(Math.random() * text.length + 1)-1];
    $('.message').stop();
    $('.message').html(text).fadeTo(200, 1);
    if (timeout === null) timeout = 5000;
    hideMessage(timeout);
}

function hideMessage(timeout){
    $('.message').stop().css('opacity',1);
    if (timeout === null) timeout = 5000;
    $('.message').delay(timeout).fadeTo(200, 0);
}

