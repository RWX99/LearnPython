from flask import Flask, render_template, jsonify, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'asdfghjkl'

DATA_DICT = {
    1: {'name': '陈硕', 'age': 73},
    2: {'name': '汪洋', 'age': 84},
}


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')  # render
        # return jsonify({'code': 1000, 'data': [1, 2, 3]}) # JsonResponse
    user = request.form.get('user')
    pwd = request.form.get('pwd')
    if user == 'changxin' and pwd == 'dsb':
        session['xxx'] = 'changxin'
        return redirect('/index')
    error = '用户名或密码错误'
    return render_template('login.html', error=error)


@app.route('/index', endpoint='idx')
def index():
    data_dict = DATA_DICT
    return render_template('index.html', data_dict=data_dict)


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    nid = int(request.args.get('nid'))

    if request.method == 'GET':
        info = DATA_DICT[nid]
        return render_template('edit.html', info=info)
    user = request.form.get('user')
    age = request.form.get('age')
    DATA_DICT[nid]['name'] = user
    DATA_DICT[nid]['age'] = age
    return redirect(url_for('idx'))


@app.route('/del/<int:nid>')
def delete(nid):
    del DATA_DICT[nid]
    # return redirect('/index')
    return redirect(url_for('idx'))


if __name__ == '__main__':
    app.run()
