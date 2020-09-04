package com.example.facialeditapp

import android.app.Activity
import android.graphics.Bitmap
import android.os.AsyncTask
import android.view.View
import android.widget.TextView
import java.io.*
import java.net.HttpURLConnection
import java.net.URL
import java.util.Base64


class UploadImageHttpRequest : AsyncTask<Param, Void, String> {
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
        // 戻り値をViewにセット
        val textView = mActivity!!.findViewById<View>(R.id.textView) as TextView
        textView.text = string
    }
}
