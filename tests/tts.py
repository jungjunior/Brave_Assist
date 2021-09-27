import pyttsx3
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


engine.say("Teste de voz, hey, hey, testando... Cordas vocais funcionando!")
engine.runAndWait()
engine.stop()
