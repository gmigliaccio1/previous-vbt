#include <M5Unified.h>
#include <LittleFS.h>
#include <vector>
#include <Arduino.h>

constexpr uint32_t SAMPLE_PERIOD_US = 5000;
constexpr uint32_t MAX_SECONDS = 120;
constexpr uint32_t MAX_SAMPLES = (1000000UL * MAX_SECONDS) / SAMPLE_PERIOD_US;

struct __attribute__((packed)) Sample {
  uint32_t t_us;
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};

enum class State{
  IDLE,
  RECORDING
};

Sample* buf = nullptr;
size_t count = 0;
uint32_t t0 = 0, nextTick = 0, lateCount = 0;
State state = State::IDLE;


static int16_t toI16(float v, float scale){
  float s = v*scale;
  if(s>32767){
    s=32767;
  }else if(s<-32768){
    s=-32768;
  }
  return (int16_t)lroundf(s);
}

void show(const char *l1, const char*l2 = "", const char* l3 = ""){
  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(0, 0);
  M5.Display.println(l1);
  M5.Display.println(l2);
  M5.Display.println(l3);
}

void startRecording(){
  count = 0;
  lateCount = 0;
  show("REC", "press to", "stop");
  t0 = micros();
  nextTick = t0;
  state = State::RECORDING;
}

void stopRecording(){
  state = State::IDLE;
  show("SAVING...");
  char name[24];
  for(int i=1; i<1000; i++){
    snprintf(name, sizeof(name), "/set_%03d.bin", i);
    if(!LittleFS.exists(name)){
      break;
    }
  }
  size_t expected = count * sizeof(Sample);
  size_t written = 0;
  File f = LittleFS.open(name, "w");
  if(f){
    written = f.write((const uint8_t*)buf, expected);
    f.close();
  }
  char l2[24], l3[24];
  snprintf(l2, sizeof(l2), "%.1f s", count * SAMPLE_PERIOD_US / 1e6);
  snprintf(l3, sizeof(l3), "late %lu", (unsigned long)lateCount);
  show(written == expected ? name + 1 : "SAVE FAILED", l2, l3);
}

void sampleOnce(){
  uint32_t now = micros();
  int32_t lateness = (int32_t)(now - nextTick);
  if(lateness < 0){
    return;
  }
  if(lateness > (int32_t)(2*SAMPLE_PERIOD_US)){
    lateCount++;
    nextTick = now;
  }
  nextTick += SAMPLE_PERIOD_US;

  if(!M5.Imu.update()){
    return;
  }
  auto d = M5.Imu.getImuData();
  Sample &s  = buf[count++];
  s.t_us = now - t0;
  s.ax = toI16(d.accel.x, 1000.0f);
  s.ay = toI16(d.accel.y, 1000.0f);
  s.az = toI16(d.accel.z, 1000.0f);
  s.gx = toI16(d.gyro.x, 10.0f);
  s.gy = toI16(d.gyro.y, 10.0f);
  s.gz = toI16(d.gyro.z, 10.0f);

  if(count >= MAX_SAMPLES){
    stopRecording();
  }
}

void dumpFile(String name){
  if(!name.startsWith("/")){
    name = "/" + name;
  }
  File f = LittleFS.open(name, "r");
  if(!f){
    Serial.println("ERR");
    return;
  }
  Serial.println("BEGIN");
  Serial.println("t_s,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
  Sample s;
  while (f.read((uint8_t*)&s, sizeof(s)) == sizeof(s)) {
    Serial.printf("%.6f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f\n",
                  s.t_us / 1e6, s.ax / 1000.0, s.ay / 1000.0, s.az / 1000.0,
                  s.gx / 10.0, s.gy / 10.0, s.gz / 10.0);
  }
  f.close();
  Serial.println("END");
}

void handleSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd == "ls") {
    File root = LittleFS.open("/");
    for (File f = root.openNextFile(); f; f = root.openNextFile()) {
      Serial.printf("FILE %s %u\n", f.name(), (unsigned)f.size());
    }
    Serial.println("END");
  } else if (cmd.startsWith("dump ")) {
    dumpFile(cmd.substring(5));
  } else if (cmd == "rmall") {
    std::vector<String> names;
    File root = LittleFS.open("/");
    for (File f = root.openNextFile(); f; f = root.openNextFile()) {
      names.push_back(String("/") + f.name());
    }
    for (auto& n : names) LittleFS.remove(n);
    Serial.println("END");
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);
  M5.Display.setTextSize(2);

  buf = (Sample*)ps_malloc(MAX_SAMPLES * sizeof(Sample));
  if (!buf) { show("NO PSRAM"); while (true) delay(1000); }
  if (!LittleFS.begin(true)) { show("FS ERROR"); while (true) delay(1000); }
  char batLine[24];
  snprintf(batLine, sizeof(batLine), "bat %d%%%s",
         M5.Power.getBatteryLevel(),
         M5.Power.isCharging() ? " chg" : "");
  show("READY", "press btn", batLine);
}

void loop() {
  static uint32_t lastPoll = 0;
  if (millis() - lastPoll >= 20) {
    lastPoll = millis();
    M5.update();
    if (M5.BtnA.wasPressed()) {
      if (state == State::IDLE) startRecording();
      else stopRecording();
    }
  }
  if (state == State::RECORDING) {
    sampleOnce();
  } else {
    handleSerial();
    delay(2);
  }
}