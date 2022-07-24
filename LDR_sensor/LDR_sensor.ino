void setup() {
  Serial.begin(9600);
  pinMode(13,OUTPUT);

}

void loop() {
  int LDR_out = analogRead(A1);
  if(LDR_out >300){
    digitalWrite(13,HIGH);
  }
  else{
    digitalWrite(13,LOW);
  }

}
