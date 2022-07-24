void setup() {
  Serial.begin(9600);
  pinMode(12,INPUT);
  pinMode(13,OUTPUT);

}

void loop() {
  int IR_out = digitalRead(12);
  if(IR_out == 0){ // Triggers the LEÐ ON if IR senor detects obstacle
    digitalWrite(13,HIGH);
  }
  else{
    digitalWrite(13,LOW);
  }

}
