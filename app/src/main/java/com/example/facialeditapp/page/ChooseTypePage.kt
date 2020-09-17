package com.example.facialeditapp.page

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.AsyncTask
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.util.Log
import android.view.View
import android.widget.ImageView
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
            val bmp: Bitmap = getBitmapFromUri(imageUri)

            Log.i("bmp", "width:" + bmp.width + " " + "height:" + bmp.height)

            setOriginalBmpSize(bmp)

            val resizedBmp = resizeBitmap(bmp)

            Log.i("resizedBmp", "width:" + resizedBmp.width + " " + "height:" + resizedBmp.height)

            UploadImageHttpRequest(this).execute(
                Param(
                    "http://192.168.10.6:9004/image",
                    resizedBmp
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
        private var mActivity: Activity? = null

        constructor(activity: Activity?) {
            mActivity = activity
        }

        override fun doInBackground(vararg params: Param): String? {
            val param: Param = params[0]
            var connection: HttpURLConnection? = null
            val sb = StringBuilder()
            try {
                //画像をjpeg形式でstreamに保存
                val jpg = ByteArrayOutputStream()
                param.bmp?.compress(Bitmap.CompressFormat.JPEG, 100, jpg)

                //streamをbyte配列に変換し, Base64でエンコード
                val encodedJpg: String = Base64.getEncoder().encodeToString(jpg.toByteArray())

                //jsonにエンコードした画像データを埋め込む
                val json: String = String.format("{ \"image\":\"%s\" } ", encodedJpg)

                val url = URL(param.uri)
                connection = url.openConnection() as HttpURLConnection
                connection.connectTimeout = 3000 //接続タイムアウトを設定する。
                connection.readTimeout = 3000 //レスポンスデータ読み取りタイムアウトを設定する。
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

            val jsonObject = JSONObject(string)

            Log.w("ここ", jsonObject.get("img").toString())

            val intent = Intent(this@ChooseTypePage, SaveImagePage::class.java)
            intent.putExtra(OUTPUT, jsonObject.get("img").toString())
            startActivity(intent)
        }
    }
}