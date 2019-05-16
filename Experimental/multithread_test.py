from threading import Thread
import time
class IlMioThread (Thread):
   def __init__(self, nome, durata):
      Thread.__init__(self)
      self.nome = nome
      self.durata = durata
   def run(self):
      print ("Thread '" + self.name + "' avviato")
      time.sleep(self.durata)
      print ("Thread '" + self.name + "' inizio calcolo")
      j = 0
      for i in range(10000000):
          j = i + j
      print ("Thread '" + self.name + "' terminato " + str(j))

from random import randint
# Creazione dei thread
start = time.time()
thread1 = IlMioThread("Thread#1", 0)
thread2 = IlMioThread("Thread#2", 0)
thread3 = IlMioThread("Thread#3", 0)
thread4 = IlMioThread("Thread#4", 0)
thread5 = IlMioThread("Thread#5", 0)
thread6 = IlMioThread("Thread#6", 0)
thread7 = IlMioThread("Thread#7", 0)
thread8 = IlMioThread("Thread#8", 0)
thread9 = IlMioThread("Thread#9", 0)
thread10= IlMioThread("Thread#10", 0)
# Avvio dei thread
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
thread8.start()
thread9.start()
thread10.start()
# Join
thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()
thread6.join()
thread7.join()
thread8.join()
thread9.join()
thread10.join()
end = time.time()
print("Tempo di esecuzione: ", end - start)
# Fine dello script
print("Fine")

start = time.time()
j = 0
for i in range(10000000):
    j = i + j
print ("Thread '" + 'xxx' + "' terminato" + str(j))
end = time.time()
print("Tempo di esecuzione: ", end - start)
