package com.example.facialeditapp.page

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.AsyncTask
import android.os.Bundle
import android.os.Handler
import android.os.ParcelFileDescriptor
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import com.example.facialeditapp.Param
import com.example.facialeditapp.R
import org.json.JSONObject
import java.io.*
import java.net.HttpURLConnection
import java.net.URL
import java.util.*


class ChooseTypePage : AppCompatActivity() {
    companion object {
        val OUTPUT = "OUTPUT"
        var originalHeight = 256
        var originalWidth = 256
    }
    private var imageUri: Uri = Uri.EMPTY

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.choose_type_page)
        val myToolbar: Toolbar = findViewById(R.id.my_toolbar)
        setSupportActionBar(myToolbar)

        imageUri = intent.extras?.get(SelectImagePage.IMAGE_URI) as Uri

        val imageView = findViewById<ImageView>(R.id.selectedImage)
        imageView.setImageURI(imageUri)
    }

    public fun clickEditButton(view: View) {
        try {
            val typeButtonGroup = findViewById<RadioGroup>(R.id.type_button_group)
            val selectedTypeButton = findViewById<RadioButton>(typeButtonGroup.checkedRadioButtonId)

            //真顔のタイプ
            var type = 1
            if(getString(R.string.egao) == selectedTypeButton?.text) {
                //笑顔のタイプ
                type = 2
            }
            Log.i("type", "type: $type")

            val bmp: Bitmap = getBitmapFromUri(imageUri)
            Log.i("bmp", "width:" + bmp.width + " " + "height:" + bmp.height)

            setOriginalBmpSize(bmp)

            val resizedBmp = resizeBitmap(bmp)
            Log.i("resizedBmp", "width:" + resizedBmp.width + " " + "height:" + resizedBmp.height)

            val handler = Handler()

            UploadImageHttpRequest( handler ).execute(
                Param(
                    "http://35.221.90.53:9004/image",
                    resizedBmp,
                    type
                )
            )
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    @Throws(IOException::class)
    private fun getBitmapFromUri(uri: Uri): Bitmap {
        val parcelFileDescriptor: ParcelFileDescriptor? = contentResolver.openFileDescriptor(uri, "r")
        val fileDescriptor: FileDescriptor? = parcelFileDescriptor?.fileDescriptor
        val image: Bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor)
        parcelFileDescriptor?.close()
        return image
    }

    private fun setOriginalBmpSize(bmp: Bitmap) {
        originalHeight = bmp.height
        originalWidth = bmp.width
    }

    private fun resizeBitmap(before: Bitmap): Bitmap {
        val height = 551
        val width  = 413

        return Bitmap.createScaledBitmap(before, width,  height,true)
    }

    inner class UploadImageHttpRequest : AsyncTask<Param, Void, String> {
        private var handler: Handler? = null
        private val progressBar: ProgressBar? = findViewById(R.id.progressBar)

        constructor(handler: Handler?) {
            this.handler = handler
        }

        override fun doInBackground(vararg params: Param): String? {
            val param: Param = params[0]
            var connection: HttpURLConnection? = null
            val sb = StringBuilder()
            try {
                val type = param.type

                //画像をjpeg形式でstreamに保存
                val jpg = ByteArrayOutputStream()
                param.bmp?.compress(Bitmap.CompressFormat.JPEG, 100, jpg)

                //streamをbyte配列に変換し, Base64でエンコード
                val encodedJpg: String = Base64.getEncoder().encodeToString(jpg.toByteArray())

                //jsonにエンコードした画像データを埋め込む
                val json: String = String.format("{ \"image\":\"%s\", \"type\":\"%d\" } ", encodedJpg, type)

                val url = URL(param.uri)
                connection = url.openConnection() as HttpURLConnection
                connection.connectTimeout = 3000 //接続タイムアウトを設定する。
                connection.readTimeout = 50000 //レスポンスデータ読み取りタイムアウトを設定する。
                connection.requestMethod = "POST" //HTTPのメソッドをPOSTに設定する。

                //ヘッダーを設定する
                connection.setRequestProperty("User-Agent", "Android")
                connection.setRequestProperty("Accept", "application/json")
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doInput = true //リクエストのボディ送信を許可する
                connection.doOutput = true //レスポンスのボディ受信を許可する
                connection.useCaches = false //キャッシュを使用しない
                connection.instanceFollowRedirects = false
                connection.connect()

                handler!!.post { progressBar?.visibility = ProgressBar.VISIBLE }

                // データを投げる
                val out: OutputStream = connection.outputStream
                val autoFlush = false
                val encoding = "UTF-8"
                val ps = PrintStream(out, autoFlush, encoding)
                ps.print(json)
                ps.close()

                // データを受け取る
                val `is`: InputStream = connection.inputStream
                val reader = BufferedReader(InputStreamReader(`is`, "UTF-8"))
                var line: String? = ""
                while (reader.readLine().also{line = it} != null) sb.append(line)
                `is`.close()

            } catch (e: IOException) {
                e.printStackTrace()
            } finally {
                connection?.disconnect()
            }
            return sb.toString()
        }

        override fun onPostExecute(string: String?) {

            handler!!.post { progressBar?.visibility = ProgressBar.INVISIBLE }

            val jsonObject = JSONObject(string)

            val intent = Intent(this@ChooseTypePage, SaveImagePage::class.java)
            intent.putExtra(OUTPUT, jsonObject.get("img").toString())
            startActivity(intent)
        }
    }
}